"""Anchorage harbor depth forecast (refill model + ConvLSTM correction) for the dredging page.

This is anchorage/anchorage_deploy.ipynb as plain functions, for the web server to call:

    from app.dredging_forecast import forecast, ForecastError
    geojson = forecast('2026-11-13')          # dict, ready for jsonify / json.dump

What forecast(date) does:
  1. reads the eHydro survey catalogue for CEPOA_AK_16_ANC (cached for a few hours)
  2. picks the inputs: the last survey 21-32 days before the date (closest to 28 days)
     plus 3 earlier ones, every 2nd survey, from the same season
  3. downloads those 4 surveys and puts them on the 10 m model grid (cached per survey)
  4. refill forecast (closed-form formula from the bundle) + ConvLSTM correction
     (3 torch models averaged, same code as the notebook)
  5. returns 10 m squares as GeoJSON, in lon/lat, with depth_ft and change_ft

Bad or impossible dates raise ForecastError (a ValueError) with a message meant for the user.
Network problems raise requests / remotezip exceptions.

Command line test (from backend/):
    python -m app.dredging_forecast 2026-11-13 out.geojson
"""
import io
import os
import pickle
import threading
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from remotezip import RemoteZip
from scipy.ndimage import uniform_filter

BUNDLE_PATH = os.environ.get(
    'DREDGING_BUNDLE',
    os.path.join(os.path.dirname(__file__), 'dredging', 'anchorage_refill_convlstm_bundle.pkl'),
)
CATALOG_URL = ('https://services7.arcgis.com/n1YM8pTrFmm7L4hs/arcgis/rest/services/'
               'eHydro_Survey_Data/FeatureServer/0/query')
CHANNEL = 'CEPOA_AK_16_ANC'
CATALOG_TTL_S = 6 * 3600          # re-read the survey list at most every 6 hours
HTTP_TIMEOUT_S = 60
MARGIN_DAYS = 14                  # how far past the training season we still forecast, with a warning
FEET_TO_M = 1200 / 3937           # US survey foot

_lock = threading.Lock()
_bundle = None
_catalog = None                   # (time fetched, DataFrame)


class ForecastError(ValueError):
    """The date cannot be forecast. The message explains why and can be shown to the user."""


# ---------------------------------------------------------------------------------------------
# 1. model bundle
# ---------------------------------------------------------------------------------------------

class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickling can run arbitrary code, so only allow what the bundle contains: numpy arrays."""
    ALLOWED = {('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', '_reconstruct'),
               ('numpy', 'ndarray'), ('numpy', 'dtype')}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'refusing to load {module}.{name} from the model bundle')
        if module == 'numpy._core.multiarray':         # saved with numpy 2, server may run numpy 1
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = 'numpy.core.multiarray'
        return super().find_class(module, name)


def load_bundle():
    """The trained model (loaded once per process)."""
    global _bundle
    with _lock:
        if _bundle is None:
            with open(BUNDLE_PATH, 'rb') as f:
                _bundle = _NumpyOnlyUnpickler(f).load()
        return _bundle


# ---------------------------------------------------------------------------------------------
# 2. survey catalogue and choice of inputs
# ---------------------------------------------------------------------------------------------

def load_catalog():
    """Usable condition surveys (IS + spring full surveys), one per date, with a season number."""
    global _catalog
    with _lock:
        if _catalog is not None and time.time() - _catalog[0] < CATALOG_TTL_S:
            return _catalog[1]
    params = {
        'where': f"channelareaidfk = '{CHANNEL}' AND surveytype = 'CS'",
        'outFields': 'surveydatestart,sourcedatalocation',
        'returnGeometry': 'false',
        'f': 'json',
    }
    response = requests.get(CATALOG_URL, params=params, timeout=HTTP_TIMEOUT_S)
    response.raise_for_status()
    surveys = pd.DataFrame([f['attributes'] for f in response.json()['features']])
    surveys['date'] = pd.to_datetime(surveys['surveydatestart'], unit='ms').dt.normalize()
    surveys['url'] = surveys['sourcedatalocation']
    surveys['name'] = surveys['url'].str.split('/').str[-1].str.replace('.ZIP', '', case=False)
    surveys['kind'] = surveys['name'].str.extract(r'_CS_([A-Za-z]+)')[0].str.upper().fillna('other')

    spring = (surveys.kind == 'other') & surveys.date.dt.month.isin([4, 5])
    usable = surveys[(surveys.kind == 'IS') | spring].sort_values('date')
    usable = usable.drop_duplicates('date', keep='last').reset_index(drop=True)
    gap = usable.date.diff().dt.days
    usable['season'] = (gap.isna() | (gap > 60)).cumsum()
    with _lock:
        _catalog = (time.time(), usable)
    return usable


def forecast_windows():
    """Date ranges that can be forecast from the current survey list, newest last.
    Each entry: {'from': 'YYYY-MM-DD', 'to': 'YYYY-MM-DD', 'last_survey': 'YYYY-MM-DD'}."""
    bundle, usable = load_bundle(), load_catalog()
    lo, hi = bundle['lead_days']
    span = bundle['STEP'] * (bundle['K'] - 1)
    windows = []
    for i in range(span, len(usable)):
        if usable.season[i - span] == usable.season[i]:
            d = usable.date[i]
            windows.append({'from': str((d + pd.Timedelta(days=lo)).date()),
                            'to': str((d + pd.Timedelta(days=hi)).date()),
                            'last_survey': str(d.date())})
    return windows


def pick_inputs(target, usable, bundle, origin=None):
    """Rows of `usable` to feed the model for `target`, plus a warning (or None).
    Same rules as training: last input 21-32 days before the target, 3 more every STEP surveys.
    `origin` (a Timestamp) forces the last input to be the survey on that date."""
    lo, hi = bundle['lead_days']
    first_doy, last_doy = bundle['target_day_of_year']
    STEP, K = bundle['STEP'], bundle['K']

    doy = target.dayofyear - 1                         # days since 1 January, as in training
    if not first_doy - MARGIN_DAYS <= doy <= last_doy + MARGIN_DAYS:
        raise ForecastError(
            f'{target.date()} is too far outside the season the model was trained on. Over the winter '
            'there are no regular surveys and the model has never seen that period.')
    warning = None
    if not first_doy <= doy <= last_doy:
        warning = (f'{target.date()} is outside the dates the model was trained on; the seasonal part '
                   'of the forecast is extrapolated, so treat it with extra care.')

    near = usable[(usable.date >= target - pd.Timedelta(days=hi)) &
                  (usable.date <= target - pd.Timedelta(days=lo))]
    if near.empty:
        raise ForecastError(f'There is no survey {lo}-{hi} days before {target.date()} to forecast from.')
    if origin is not None:
        near = near[near.date == origin]
        if near.empty:
            raise ForecastError(f'No usable survey on {origin.date()} that is {lo}-{hi} days before {target.date()}.')
    last_i = (near.date - (target - pd.Timedelta(days=28))).abs().idxmin()
    input_i = [last_i - STEP * k for k in range(K - 1, -1, -1)]
    if input_i[0] < 0 or usable.season[input_i[0]] != usable.season[last_i]:
        raise ForecastError(f'The survey of {usable.date[last_i].date()} is too early in its season: '
                            f'there are not {K - 1} earlier surveys in the same season to go with it.')
    return input_i, warning


# ---------------------------------------------------------------------------------------------
# 3. download a survey and put it on the grid
# ---------------------------------------------------------------------------------------------

def _read_points(url):
    with RemoteZip(url, support_suffix_range=False, timeout=HTTP_TIMEOUT_S) as z:
        xyz = [i for i in z.infolist() if '/' not in i.filename and i.filename.upper().endswith('.XYZ')]
        full = [i for i in xyz if i.filename.upper().endswith('_FULL.XYZ')]
        chosen = full[0] if full else max(xyz, key=lambda i: i.file_size)
        text = z.read(chosen.filename).decode('latin-1')
    data = '\n'.join(line for line in text.splitlines() if '==' not in line and line.strip()).replace(',', ' ')
    points = pd.read_csv(io.StringIO(data), sep=r'\s+', header=None, names=['x', 'y', 'elevation'])
    points = pd.DataFrame({'x': points.x * FEET_TO_M, 'y': points.y * FEET_TO_M, 'depth': -points.elevation})
    points = points.drop_duplicates()
    points = points[(points.depth >= -30) & (points.depth <= 120)]
    return points[points.x.between(504_000, 507_000) & points.y.between(804_500, 808_800)]


@lru_cache(maxsize=32)
def survey_map(url):
    """Mean depth (ft) per 10 m cell on the model grid, NaN where not measured. Cached per survey."""
    G = load_bundle()['grid']
    H, W = G['shape']
    points = _read_points(url)
    rows = (points.y // G['cell_m']).astype(int) - G['row0']
    cols = (points.x // G['cell_m']).astype(int) - G['col0']
    inside = rows.between(0, H - 1) & cols.between(0, W - 1)
    cells = points[inside].assign(r=rows[inside], c=cols[inside]).groupby(['r', 'c']).depth.mean()
    grid = np.full((H, W), np.nan, np.float32)
    grid[cells.index.get_level_values(0), cells.index.get_level_values(1)] = cells.values
    grid.setflags(write=False)                         # shared through the cache: keep it unchanged
    return grid


# ---------------------------------------------------------------------------------------------
# 4. the model: refill formula + ConvLSTM correction (numpy)
# ---------------------------------------------------------------------------------------------

def refill_forecast(input_maps, input_days, lead, target, bundle):
    """Refill model: predicted depth change (ft) per cell from the last input to the target."""
    last = input_maps[-1]
    D = np.nan_to_num(last)
    measured = np.isfinite(last)

    # p: how much shallower/deeper a cell is than its 5x5 neighbours (pit pull)
    sums = uniform_filter(D, size=5, mode='constant') * 25
    counts = uniform_filter(measured.astype(np.float32), size=5, mode='constant') * 25
    p = np.where(measured & (counts - measured > 0), (sums - D) / np.maximum(counts - measured, 1) - D, 0)

    # m: per-cell trend (ft/day) fitted through the 4 inputs
    ok = np.isfinite(input_maps)
    n = ok.sum(0)
    t_mean = (ok * input_days[:, None, None]).sum(0) / np.maximum(n, 1)
    y_mean = np.where(ok, input_maps, 0).sum(0) / np.maximum(n, 1)
    dt = np.where(ok, input_days[:, None, None] - t_mean, 0)
    m = np.where(n >= 2, (dt * np.where(ok, input_maps - y_mean, 0)).sum(0) / np.maximum((dt ** 2).sum(0), 1e-6), 0)

    theta = (target.dayofyear - 1) * 2 * np.pi / 365   # as in training: days since 1 January
    a, b, c = bundle['refill_abc']
    r = bundle['refill_r']
    return (a * p + b * m * lead + c * (D - 45) * lead
            + lead * (r[..., 0] + r[..., 1] * np.sin(theta) + r[..., 2] * np.cos(theta))).astype(np.float32)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel_size=3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Corrector(nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x, static):              # x (1, 4, 2, H, W), static (1, 1, H, W)
        h = torch.zeros(1, self.cell.hidden, *x.shape[-2:])
        c = torch.zeros_like(h)
        for t in range(x.shape[1]):
            h, c = self.cell(torch.cat([x[:, t], static], dim=1), h, c)
        return self.head(h)[:, 0]


_nets = None


def load_nets():
    """The trained ConvLSTMs from the bundle, built once per process."""
    global _nets
    bundle = load_bundle()
    with _lock:
        if _nets is None:
            nets = []
            for weights in bundle['members']:
                net = Corrector(bundle['in_channels'], bundle['hidden'])
                net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()})
                net.eval()
                nets.append(net)
            _nets = nets
        return _nets


def correction_forecast(input_maps, refill, bundle):
    """Average correction (ft) of the ensemble, added on top of the refill forecast."""
    measured = np.isfinite(input_maps)
    x = np.stack([np.where(measured, (input_maps - bundle['depth_mean']) / bundle['depth_std'], 0), measured],
                 axis=1)                           # (T, 2, H, W)
    mu, sd = bundle['refill_map_norm']
    static = np.clip((refill - mu) / sd, -5, 5)[None, None]
    x = torch.tensor(x[None], dtype=torch.float32)
    static = torch.tensor(static, dtype=torch.float32)
    with torch.no_grad():
        runs = [net(x, static)[0].numpy() for net in load_nets()]
    return np.mean(runs, axis=0) * bundle['res_std']


# ---------------------------------------------------------------------------------------------
# 5. GeoJSON
# ---------------------------------------------------------------------------------------------

def to_geojson(depth, change, show, bundle, properties):
    """10 m squares where `show` is True, in lon/lat (EPSG:4326)."""
    from pyproj import Transformer
    G = bundle['grid']
    H, W = G['shape']
    cell = G['cell_m']
    # corners of every grid cell, transformed once: node (i, j) is the lower-left corner of cell (i, j)
    xs = (G['col0'] + np.arange(W + 1)) * cell
    ys = (G['row0'] + np.arange(H + 1)) * cell
    X, Y = np.meshgrid(xs, ys)
    to_lonlat = Transformer.from_crs('EPSG:26934', 'EPSG:4326', always_xy=True)   # Alaska State Plane 4 (m)
    lon, lat = to_lonlat.transform(X, Y)

    features = []
    for i, j in zip(*np.where(show)):
        corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon',
                         'coordinates': [[[float(lon[a, b]), float(lat[a, b])] for a, b in corners]]},
            'properties': {'depth_ft': round(float(depth[i, j]), 2), 'change_ft': round(float(change[i, j]), 2)},
        })
    return {'type': 'FeatureCollection', 'properties': properties, 'features': features}


# ---------------------------------------------------------------------------------------------
# 6. everything together
# ---------------------------------------------------------------------------------------------

def forecast(date, origin=None):
    """GeoJSON forecast of the harbor depth on `date` ('YYYY-MM-DD' or anything pandas understands).
    `origin` optionally forces the last input survey (used for the website's historical frames)."""
    depth, change, show, properties = forecast_arrays(date, origin)
    return to_geojson(depth, change, show, load_bundle(), properties)


def forecast_arrays(date, origin=None):
    """The forecast as grids: (depth, change, show, properties). `show` marks the cells that are
    exported, in the same order as the GeoJSON features (np.where(show))."""
    try:
        target = pd.Timestamp(date).normalize()
        origin = None if origin is None else pd.Timestamp(origin).normalize()
    except (ValueError, TypeError):
        raise ForecastError(f'Not a valid date: {date!r}. Use YYYY-MM-DD.')
    bundle, usable = load_bundle(), load_catalog()
    input_i, warning = pick_inputs(target, usable, bundle, origin)
    last_i = input_i[-1]
    lead = (target - usable.date[last_i]).days

    input_maps = np.stack([survey_map(url) for url in usable.url[input_i]])          # (4, H, W)
    input_days = np.array([(d - usable.date[last_i]).days for d in usable.date[input_i]], np.float32)

    refill = refill_forecast(input_maps, input_days, lead, target, bundle)
    correction = correction_forecast(input_maps, refill, bundle)

    last = input_maps[-1]
    show = bundle['core'] & np.isfinite(last)          # only where the last input measured the channel
    change = refill + correction                       # + = deeper
    depth = last + change

    properties = {
        'predicted_for': str(target.date()),
        'from_survey': str(usable.date[last_i].date()),
        'lead_days': lead,
        'inputs': [str(d.date()) for d in usable.date[input_i]],
        'model': bundle['model'],
        'warning': warning,
    }
    return depth, change, show, properties


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) not in (2, 3):
        sys.exit('usage: python -m app.dredging_forecast YYYY-MM-DD [out.geojson]')
    started = time.time()
    result = forecast(sys.argv[1])
    print(json.dumps(result['properties'], indent=1))
    print(len(result['features']), 'squares in', round(time.time() - started, 1), 's')
    if len(sys.argv) == 3:
        with open(sys.argv[2], 'w') as f:
            json.dump(result, f)
        print('saved', sys.argv[2])
