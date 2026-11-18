(() => {
  'use strict';

  const el = id => document.getElementById(id);
  const set = (id, value) => { el(id).textContent = value; };
  const number = value => new Intl.NumberFormat('en-US').format(value);
  const feet = (value, signed = false) => `${signed && value > 0 ? '+' : ''}${value.toFixed(2)} ft`;
  const date = value => new Intl.DateTimeFormat('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'
  }).format(new Date(`${value}T00:00:00Z`));
  const palettes = {
    depth: { min: 0, max: 80, binWidth: 10, stops: ['#f8d77b', '#8ad1c2', '#3296b2', '#23558e', '#17294c'] },
    change: { min: -3, max: 3, binWidth: 0.5, stops: ['#b3263e', '#e88372', '#f5f0df', '#80bfd0', '#2465a0'] }
  };
  // Keep the platform's hybrid basemap free of business and transit icons.
  const HIDDEN_MAP_FEATURES = [
    { featureType: 'poi.business', stylers: [{ visibility: 'off' }] },
    { featureType: 'transit', elementType: 'labels.icon', stylers: [{ visibility: 'off' }] }
  ];
  const state = {
    manifest: null, map: null, overlay: null, hotspots: null, popup: null, prepared: null,
    mode: 'depth', fitted: false, request: 0, controller: null, cache: new Map(), mapsTimer: null
  };
  const paletteRGB = Object.fromEntries(Object.entries(palettes).map(([metric, palette]) => [
    metric, palette.stops.map(hex => [1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16)))
  ]));

  function colorForMetric(metric, value) {
    const palette = palettes[metric];
    const position = Math.max(0, Math.min(1, (value - palette.min) / (palette.max - palette.min))) * (palette.stops.length - 1);
    const index = Math.min(palette.stops.length - 2, Math.floor(position));
    const fraction = position - index;
    const [start, end] = paletteRGB[metric].slice(index, index + 2);
    return `rgb(${start.map((channel, i) => Math.round(channel + (end[i] - channel) * fraction)).join(',')})`;
  }

  function renderLegend() {
    const palette = palettes[state.mode];
    el('legendGradient').style.background = `linear-gradient(to right, ${palette.stops.join(',')})`;
    set('legendMin', String(palette.min));
    set('legendMax', String(palette.max));
    set('legendCaption', state.mode === 'depth' ? 'Depth (ft)' : 'Change (ft)');
    set('legend-description', state.mode === 'depth'
      ? 'Yellow: shallow · dark blue: deep. Fixed 0–80 ft scale.'
      : 'Red: shallower · blue: deeper. Zero-centered, fixed −3 to +3 ft scale.');
  }

  function clearOverlay() {
    state.popup?.close();
    state.overlay?.setMap(null);
    state.overlay = null;
    state.hotspots?.setMap(null);
    state.hotspots = null;
  }

  function clearForecast(message) {
    clearOverlay();
    state.prepared = null;
    const chart = el('distribution-chart');
    chart.replaceChildren();
    chart.setAttribute('hidden', '');
    set('distribution-empty', message);
    el('distribution-empty').hidden = false;
    for (const id of ['cell-count', 'distribution-area', 'distribution-min', 'distribution-median', 'distribution-mean', 'distribution-max', 'model-score']) set(id, '—');
    el('download-geojson').removeAttribute('href');
    el('download-geojson').removeAttribute('download');
    el('download-geojson').setAttribute('aria-disabled', 'true');
  }

  function summarize(values, metric) {
    const palette = palettes[metric];
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const bins = Array.from({ length: Math.round((palette.max - palette.min) / palette.binWidth) }, () => 0);
    for (const value of values) {
      // Outliers stay in the end bars, so every prediction cell is counted.
      bins[Math.max(0, Math.min(bins.length - 1, Math.floor((value - palette.min) / palette.binWidth)))] += 1;
    }
    return {
      min: sorted[0], max: sorted[sorted.length - 1],
      median: sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2,
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      bins
    };
  }

  function prepareCollection(collection, frame) {
    if (collection.type !== 'FeatureCollection' || !Array.isArray(collection.features) || !collection.features.length ||
        collection.properties?.from_survey !== frame.from_survey || collection.properties?.predicted_for !== frame.predicted_for) {
      throw new Error('The forecast file does not match the selected survey and prediction dates.');
    }
    const values = { depth: [], change: [] };
    const hotspotGroups = new Map();
    const bounds = { north: -90, south: 90, east: -180, west: 180 };
    for (const feature of collection.features) {
      const ring = feature.geometry?.coordinates?.[0];
      if (feature.type !== 'Feature' || feature.geometry?.type !== 'Polygon' || !Array.isArray(ring) || ring.length !== 5 ||
          !Number.isFinite(feature.properties?.depth_ft) || !Number.isFinite(feature.properties?.change_ft) ||
          !ring.every(point => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite) && Math.abs(point[0]) <= 180 && Math.abs(point[1]) <= 90) ||
          ring[0][0] !== ring[4][0] || ring[0][1] !== ring[4][1]) {
        throw new Error('The forecast file contains an invalid prediction cell.');
      }
      for (const [longitude, latitude] of ring) {
        bounds.north = Math.max(bounds.north, latitude);
        bounds.south = Math.min(bounds.south, latitude);
        bounds.east = Math.max(bounds.east, longitude);
        bounds.west = Math.min(bounds.west, longitude);
      }
      values.depth.push(feature.properties.depth_ft);
      values.change.push(feature.properties.change_ft);
      if (feature.properties.change_ft <= -1) {
        const longitude = ring.slice(0, 4).reduce((sum, point) => sum + point[0], 0) / 4;
        const latitude = ring.slice(0, 4).reduce((sum, point) => sum + point[1], 0) / 4;
        const key = `${latitude.toFixed(3)},${longitude.toFixed(3)}`;
        const group = hotspotGroups.get(key) || { latitude: 0, longitude: 0, count: 0 };
        group.latitude += latitude; group.longitude += longitude; group.count += 1;
        hotspotGroups.set(key, group);
      }
    }
    // Compute both summaries once when a file is loaded, never per map feature.
    const hotspots = [...hotspotGroups.values()].map(group => ({ count: group.count, latitude: group.latitude / group.count, longitude: group.longitude / group.count }));
    return { collection, bounds, hotspots, metrics: { depth: summarize(values.depth, 'depth'), change: summarize(values.change, 'change') } };
  }

  function renderDistribution() {
    const metric = state.mode;
    const label = metric === 'depth' ? 'Depth' : 'Change';
    set('distTitle', `${label} distribution`);
    if (!state.prepared) return;
    const { collection, metrics } = state.prepared;
    const count = collection.features.length;
    const summary = metrics[metric];
    const palette = palettes[metric];
    const bins = summary.bins;
    set('cell-count', number(count));
    set('distribution-area', `${(count * state.manifest.cell_metres ** 2 / 10000).toFixed(2)} ha`);
    for (const field of ['min', 'median', 'mean', 'max']) set(`distribution-${field}`, feet(summary[field], metric === 'change'));

    const binLabel = index => index === 0 ? `Less than ${palette.min + palette.binWidth} ft`
      : index === bins.length - 1 ? `${palette.max - palette.binWidth} ft or more`
      : `${palette.min + index * palette.binWidth} to less than ${palette.min + (index + 1) * palette.binWidth} ft`;
    const svg = el('distribution-chart');
    svg.replaceChildren();
    const add = (tag, attributes, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
      if (text !== undefined) node.textContent = text;
      svg.append(node);
      return node;
    };
    add('title', { id: 'distribution-chart-title' }, `${label} distribution in feet across ${number(count)} prediction cells`);
    add('desc', { id: 'distribution-chart-description' }, `Horizontal axis: ${label.toLowerCase()} in feet. Vertical axis: cell count. ${bins.map((n, i) => `${binLabel(i)}: ${number(n)} cells`).join('; ')}. ${metric === 'change' ? 'The dashed line marks zero change.' : ''}`);
    const left = 43, right = 290, top = 24, bottom = 164;
    const highest = Math.max(...bins);
    const magnitude = 10 ** Math.floor(Math.log10(highest));
    const ceiling = Math.ceil(highest / magnitude) * magnitude;
    const x = value => left + (value - palette.min) / (palette.max - palette.min) * (right - left);
    const y = value => bottom - value / ceiling * (bottom - top);
    add('text', { x: left, y: 12, class: 'distribution-axis-label' }, 'Cell count');
    for (const tick of ceiling === 1 ? [0, 1] : [0, Math.floor(ceiling / 2), ceiling]) {
      add('line', { x1: left, y1: y(tick), x2: right, y2: y(tick), class: 'distribution-grid' });
      add('text', { x: left - 6, y: y(tick) + 4, 'text-anchor': 'end', class: 'distribution-tick' }, number(tick));
    }
    bins.forEach((n, index) => {
      const start = palette.min + index * palette.binWidth;
      const bar = add('rect', {
        x: x(start) + 1, y: y(n), width: (right - left) / bins.length - 2, height: bottom - y(n),
        fill: colorForMetric(metric, start + palette.binWidth / 2), class: 'distribution-bar', 'data-count': n
      });
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${binLabel(index)}: ${number(n)} cells`;
      bar.append(title);
    });
    if (metric === 'change') add('line', { x1: x(0), y1: top, x2: x(0), y2: bottom, class: 'distribution-zero' });
    for (let i = 0; i <= 4; i += 1) {
      const tick = palette.min + (palette.max - palette.min) * i / 4;
      add('text', { x: x(tick), y: bottom + 17, 'text-anchor': 'middle', class: 'distribution-tick' }, String(tick).replace('-', '−'));
    }
    add('text', { x: (left + right) / 2, y: 201, 'text-anchor': 'middle', class: 'distribution-axis-label' }, `${label} (ft)`);
    el('distribution-empty').hidden = true;
    svg.removeAttribute('hidden');
  }

  function predictionContent(feature) {
    const content = document.createElement('div');
    content.className = 'dredging-popup';
    const list = document.createElement('dl');
    const metadata = state.prepared.collection.properties;
    const rows = [
      ['Predicted Depth', feet(feature.getProperty('depth_ft'))],
      ['Depth Change', feet(feature.getProperty('change_ft'), true)],
      ['From Survey', date(metadata.from_survey)],
      ['Predicted For', date(metadata.predicted_for)]
    ];
    for (const [label, value] of rows) {
      const row = document.createElement('div');
      const term = document.createElement('dt');
      const detail = document.createElement('dd');
      term.textContent = label;
      detail.textContent = value;
      row.append(term, detail);
      list.append(row);
    }
    content.append(list);
    return content;
  }

  function mapStyle(feature) {
    const color = colorForMetric(state.mode, feature.getProperty(`${state.mode}_ft`));
    return { fillColor: color, fillOpacity: 0.86, strokeColor: color, strokeWeight: 0, strokeOpacity: 0, clickable: true };
  }

  function renderHotspots() {
    state.hotspots?.setMap(null);
    state.hotspots = null;
    if (!el('show-hotspots').checked || !state.map || !state.prepared) return;
    const layer = new google.maps.Data({ map: state.map });
    state.hotspots = layer;
    layer.addGeoJson({ type: 'FeatureCollection', features: state.prepared.hotspots.map(group => ({
      type: 'Feature', geometry: { type: 'Point', coordinates: [group.longitude, group.latitude] }, properties: { count: group.count }
    })) });
    layer.setStyle({ icon: { path: google.maps.SymbolPath.CIRCLE, scale: 5, strokeColor: '#8d1332', fillColor: '#ffbfbc', fillOpacity: 0.95, strokeWeight: 2 }, zIndex: 5 });
    layer.addListener('click', event => {
      const content = document.createElement('div');
      content.className = 'dredging-popup';
      content.textContent = `${number(event.feature.getProperty('count'))} nearby cells predicted to lose at least 1 ft.`;
      state.popup.setContent(content);
      state.popup.setPosition(event.latLng);
      state.popup.open({ map: state.map });
    });
  }

  function renderOverlay() {
    if (!state.map || !state.prepared) return;
    clearOverlay();
    try {
      const overlay = new google.maps.Data({ map: state.map });
      state.overlay = overlay;
      overlay.addGeoJson(state.prepared.collection);
      overlay.setStyle(mapStyle);
      renderHotspots();
      overlay.addListener('click', event => {
        state.popup.setContent(predictionContent(event.feature));
        state.popup.setPosition(event.latLng);
        state.popup.open({ map: state.map });
      });
      if (!state.fitted) {
        const padding = window.innerWidth > 680
          ? { top: 40, bottom: 40, left: 40, right: 360 }
          : 24;
        state.map.fitBounds(state.prepared.bounds, padding);
        state.fitted = true;
      }
    } catch (_) {
      clearOverlay();
      mapUnavailable('The prediction overlay could not be drawn. Forecast statistics and downloads are available.');
    }
  }

  function mapUnavailable(message = 'Satellite map unavailable. Forecast statistics and GeoJSON downloads are available below.') {
    clearTimeout(state.mapsTimer);
    set('basemap-status', message);
    el('basemap-status').hidden = false;
  }

  async function loadMap() {
    window.gm_authFailure = () => {
      clearOverlay();
      state.map = null;
      mapUnavailable('Satellite map authorization failed. Forecast statistics and GeoJSON downloads are still available.');
    };
    window.initDredgingMap = () => {
      clearTimeout(state.mapsTimer);
      try {
        state.map = new google.maps.Map(el('map'), {
          center: { lat: 61.2254, lng: -149.9051 }, zoom: 14, mapTypeId: 'hybrid',
          renderingType: 'RASTER', styles: HIDDEN_MAP_FEATURES
        });
        const title = document.createElement('h3');
        title.className = 'dredging-popup-title';
        title.textContent = 'Dredging Prediction';
        state.popup = new google.maps.InfoWindow({ headerContent: title, ariaLabel: title.textContent });
        state.map.addListener('click', () => state.popup.close());
        el('basemap-status').hidden = true;
        renderOverlay();
      } catch (_) {
        state.map = null;
        mapUnavailable();
      }
    };
    try {
      const response = await fetch('/api/config/maps-key');
      if (!response.ok) throw new Error('Map configuration unavailable.');
      const configuration = await response.json();
      const key = configuration.googleMapsApiKey;
      if (typeof key !== 'string' || !key.trim()) throw new Error('Map key unavailable.');
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&callback=initDredgingMap&loading=async`;
      script.async = true;
      script.onerror = () => mapUnavailable();
      state.mapsTimer = setTimeout(() => mapUnavailable(), 15000);
      document.head.append(script);
    } catch (_) {
      mapUnavailable();
    }
  }

  async function loadForecast() {
    const index = Number(el('forecast-frame').value);
    const frame = state.manifest.frames[index];
    const model = state.manifest.models.find(candidate => candidate.id === el('forecast-model').value);
    const url = frame.outputs[model.id];
    const request = ++state.request;
    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    el('forecast-date').value = String(index);
    el('forecast-frame').setAttribute('aria-valuetext', `${date(frame.from_survey)} survey, predicting ${date(frame.predicted_for)}`);
    set('frame-origin', date(frame.from_survey));
    set('frame-predicted-for', date(frame.predicted_for));
    el('forecast-error').hidden = true;
    el('distContent').setAttribute('aria-busy', 'true');
    clearForecast('Loading the selected forecast…');
    set('forecast-status', `Loading ${model.label}…`);
    try {
      if (!url) throw new Error('This model has no output for the selected date.');
      let prepared = state.cache.get(url);
      if (!prepared) {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) throw new Error(`Forecast could not be loaded (${response.status}).`);
        const collection = await response.json();
        if (request !== state.request) return;
        prepared = prepareCollection(collection, frame);
        state.cache.set(url, prepared);
        // Bound memory while keeping recently explored dates available instantly.
        if (state.cache.size > 6) state.cache.delete(state.cache.keys().next().value);
      }
      if (request !== state.request) return;
      state.prepared = prepared;
      renderDistribution();
      renderOverlay();
      set('model-score', model.mae_ft.toFixed(3));
      set('forecast-status', `Historical forecast · ${model.label} · ${number(prepared.collection.features.length)} cells`);
      const download = el('download-geojson');
      download.href = url;
      download.download = `anchorage-${model.id}-${frame.from_survey}.geojson`;
      download.setAttribute('aria-disabled', 'false');
    } catch (error) {
      if (error.name === 'AbortError' || request !== state.request) return;
      clearForecast('No forecast cells to summarize.');
      set('forecast-status', 'Forecast unavailable');
      set('forecast-error', `${error.message} Choose another date or reload the page.`);
      el('forecast-error').hidden = false;
    } finally {
      if (request === state.request) el('distContent').setAttribute('aria-busy', 'false');
    }
  }

  function validateManifest(manifest) {
    const validDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(`${value}T00:00:00Z`));
    if (manifest.schema_version !== 1 || !Array.isArray(manifest.models) || !manifest.models.length ||
        !Array.isArray(manifest.frames) || !manifest.frames.length || !(manifest.cell_metres > 0) ||
        !manifest.models.every(model => typeof model.id === 'string' && typeof model.label === 'string' && Number.isFinite(model.mae_ft)) ||
        !manifest.frames.every(frame => validDate(frame.from_survey) && validDate(frame.predicted_for) && frame.outputs &&
          (Date.parse(frame.predicted_for) - Date.parse(frame.from_survey)) / 86400000 === 28 &&
          manifest.models.every(model => typeof frame.outputs[model.id] === 'string' && frame.outputs[model.id].startsWith('/static/data/anchorage/')))) {
      throw new Error('The forecast index is invalid.');
    }
  }

  async function initialize() {
    renderLegend();
    document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
      state.mode = button.dataset.mode;
      document.querySelectorAll('[data-mode]').forEach(candidate => {
        candidate.classList.toggle('active', candidate === button);
        candidate.setAttribute('aria-pressed', String(candidate === button));
      });
      state.popup?.close();
      state.overlay?.setStyle(mapStyle);
      renderLegend();
      renderDistribution();
    }));
    const response = await fetch('/static/data/anchorage/manifest.json');
    if (!response.ok) throw new Error(`Forecast index could not be loaded (${response.status}).`);
    const manifest = await response.json();
    validateManifest(manifest);
    state.manifest = manifest;
    for (const model of manifest.models) {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.label;
      el('forecast-model').append(option);
    }
    manifest.frames.forEach((frame, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${date(frame.from_survey)} → ${date(frame.predicted_for)}`;
      el('forecast-date').append(option);
    });
    el('forecast-frame').max = String(manifest.frames.length - 1);
    el('forecast-frame').value = String(manifest.frames.length - 1);
    for (const id of ['forecast-model', 'forecast-date', 'forecast-frame']) el(id).disabled = false;
    el('forecast-model').addEventListener('change', loadForecast);
    el('show-hotspots').addEventListener('change', renderHotspots);
    el('forecast-frame').addEventListener('input', loadForecast);
    el('forecast-date').addEventListener('change', () => {
      el('forecast-frame').value = el('forecast-date').value;
      loadForecast();
    });
    await loadForecast();
  }

  // Forecast data and summaries load even if the map provider is unavailable.
  loadMap();
  initialize().catch(error => {
    clearForecast('Forecast preview unavailable.');
    el('distContent').setAttribute('aria-busy', 'false');
    set('forecast-status', 'Forecast preview unavailable');
    set('forecast-error', error.message);
    el('forecast-error').hidden = false;
  });
})();
