from datetime import datetime, timedelta
import urllib.parse
import json

import certifi
import requests
from flask import Blueprint, current_app, request, jsonify
from flask_cors import cross_origin
from rq.job import Job

from . import supabase
from .tasks import run_flood_job
import random

bp = Blueprint("api", __name__)
# app/routes.py

news_cache = {
    "timestamp": None,
    "articles": []
}

NEWS_QUERY = "(coral reef OR marine ecosystem OR ocean conservation) OR (aquatic sustainability OR water pollution) OR (flooding AND water) OR (drought AND water) OR (marine biodiversity OR coral bleaching)"
CACHE_TTL = timedelta(minutes=10)


def categorize_article(title: str, description: str, content: str) -> str:
    text = " ".join([title, description, content]).lower()
    
    # Coral & marine ecosystems
    if any(keyword in text for keyword in ["coral reef", "coral reefs", "coral bleaching", "marine ecosystem", "ocean health", "marine biodiversity", "marine conservation"]):
        return "coral"
    
    # Flooding with aquatic/water context
    if any(keyword in text for keyword in ["flood", "flooding", "flash flood", "river overflow", "water surge"]) and any(kw in text for kw in ["water", "aquatic", "river", "lake", "coastal", "ecosystem"]):
        return "flood"
    
    # Drought with aquatic/water context
    if any(keyword in text for keyword in ["drought", "droughts", "water scarcity", "water shortage"]) and any(kw in text for kw in ["water", "aquatic", "river", "lake", "ecosystem", "sustainability"]):
        return "drought"
    
    return "other"


def estimate_read_time(text: str) -> str:
    words = len(text.split())
    minutes = max(1, round(words / 200))
    return f"{minutes} min read"

PLACEHOLDER_ARTICLES = [
    {
        "title": "Nature: New Research on Coral Reef Ecosystems",
        "excerpt": "Placeholder summary for a Nature article related to coral reef ecosystem trends and conservation outcomes.",
        "source": "Nature",
        "date": "2026-04-10T00:00:00Z",
        "url": "https://www.nature.com/articles/s41559-026-03058-6",
        "read_time": "8 min read",
        "category": "coral",
    },
    {
        "title": "NOAA: Restoring Seven Iconic Reefs in the Florida Keys",
        "excerpt": "Placeholder summary for NOAA Fisheries updates on Mission: Recover Coral Reefs and restoration progress in the Florida Keys.",
        "source": "NOAA Fisheries",
        "date": "2026-04-08T00:00:00Z",
        "url": "https://www.fisheries.noaa.gov/southeast/habitat-conservation/restoring-seven-iconic-reefs-mission-recover-coral-reefs-florida-keys",
        "read_time": "6 min read",
        "category": "coral",
    },
    {
        "title": "Great Barrier Reef Foundation: Coral Bleaching Threats",
        "excerpt": "Placeholder summary covering the major threats of coral bleaching to the Great Barrier Reef and what it means for marine biodiversity.",
        "source": "Great Barrier Reef Foundation",
        "date": "2026-04-07T00:00:00Z",
        "url": "https://www.barrierreef.org/the-reef/threats/coral-bleaching",
        "read_time": "5 min read",
        "category": "coral",
    },
    {
        "title": "Drought Pressure Mounts for Western NC Farmers as Planting Season Begins",
        "excerpt": "Placeholder summary on severe and extreme drought conditions affecting farmers in Henderson County, North Carolina amid a significant rainfall deficit heading into the 2026 planting season.",
        "source": "News Channel 9",
        "date": "2026-04-10T00:00:00Z",
        "url": "https://newschannel9.com/news/local/drought-pressure-severe-extreme-western-north-carolina-farmers-planting-season-begins-april-2026-crops-farms-henderson-county-rainfall-deficit",
        "read_time": "4 min read",
        "category": "drought",
    },
    {
        "title": "Georgia Drought Worsens as Farmers and Waterways Feel the Strain",
        "excerpt": "Placeholder summary on worsening drought conditions across Georgia, with farmers and local waterways experiencing increasing stress as water levels continue to drop.",
        "source": "WCTV",
        "date": "2026-04-10T00:00:00Z",
        "url": "https://www.wctv.tv/2026/04/10/georgia-drought-worsens-farmers-waterways-feel-strain/",
        "read_time": "4 min read",
        "category": "drought",
    },
    {
        "title": "The West's Snow Drought Meant Record Dryness — But Also Record Flooding",
        "excerpt": "Placeholder summary on how the Western United States' severe snow drought led to both record dry conditions and paradoxical record flooding events across the region.",
        "source": "High Country News",
        "date": "2026-04-06T00:00:00Z",
        "url": "https://www.hcn.org/articles/the-wests-snow-drought-meant-record-dryness-but-also-record-flooding/",
        "read_time": "6 min read",
        "category": "drought",
    },
]

def fetch_news_articles():
    now = datetime.utcnow()
    cached_at = news_cache["timestamp"]
    if cached_at and now - cached_at < CACHE_TTL:
        return news_cache["articles"]

    api_key = current_app.config.get("NEWS_API_KEY", "")
    if not api_key:
        return PLACEHOLDER_ARTICLES

    params = {
        "q": NEWS_QUERY,
        "qInTitle": NEWS_QUERY,
        "language": "en",
        "pageSize": 40,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)

    try:
        response = requests.get(url, timeout=15, verify=certifi.where())
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        current_app.logger.error("News API fetch failed: %s", exc)
        return PLACEHOLDER_ARTICLES

    if data.get("status") != "ok":
        current_app.logger.warning("News API returned non-ok status: %s", data)
        return PLACEHOLDER_ARTICLES

    articles = []
    for article in data.get("articles", []):
        try:
            title = article.get("title") or "Untitled"
            description = article.get("description") or ""
            content = article.get("content") or ""
            source_obj = article.get("source") or {}
            source_name = source_obj.get("name", "Unknown Source") if isinstance(source_obj, dict) else "Unknown Source"
            category = categorize_article(title, description, content)
            if category == "other":
                continue
            excerpt = description or ((content[:180].rsplit(" ", 1)[0] + "...") if content else "No summary available.")
            published_at = article.get("publishedAt") or datetime.utcnow().isoformat()
            read_time = estimate_read_time(description or content)

            articles.append({
                "title": title,
                "excerpt": excerpt,
                "source": source_name,
                "date": published_at,
                "url": article.get("url", "#"),
                "read_time": read_time,
                "category": category,
            })
        except Exception as exc:
            current_app.logger.warning("Skipping malformed news article: %s", exc)
            continue

    news_cache["timestamp"] = now
    news_cache["articles"] = PLACEHOLDER_ARTICLES + articles
    return news_cache["articles"]


@bp.route("/news", methods=["GET"])
def get_news():
    """Return a small curated news feed for coral reefs, flooding, and droughts."""
    articles = fetch_news_articles()
    return jsonify({"articles": articles})


@bp.route("/flood-risk", methods=["POST"])
def start_flood_risk():
    """
    Synchronous flood risk endpoint with Redis caching.
    POST /api/flood-risk
    Body: { "lat": <float>, "lon": <float> }
    """
    data = request.get_json() or {}
    try:
        center_lat = float(data["lat"])
        center_lon = float(data["lon"])
    except (KeyError, ValueError):
        return jsonify({"error": "lat and lon are required floats"}), 400

    # Try to get from cache first
    cache_key = f"flood_risk_{center_lat:.2f}_{center_lon:.2f}"
    redis_conn = current_app.redis
    
    if redis_conn:
        try:
            cached_result = redis_conn.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                print(f"[CACHE HIT] Flood risk for {center_lat:.2f}, {center_lon:.2f}")
                return jsonify({
                    "status": "finished",
                    "result": result,
                    "cached": True,
                })
        except Exception as e:
            print(f"[CACHE READ ERROR] {e}")
            # Fall through to run model if cache read fails

    # Run the model if not cached
    try:
        result = run_flood_job(center_lat, center_lon)
    except Exception as e:
        current_app.logger.exception("Error running flood job")
        return jsonify({"error": "internal error running model"}), 500

    # Try to save to cache
    if redis_conn:
        try:
            # 24-hour TTL (86400 seconds)
            redis_conn.setex(cache_key, 86400, json.dumps(result))
            print(f"[CACHE SAVE] Flood risk for {center_lat:.2f}, {center_lon:.2f}")
        except Exception as e:
            print(f"[CACHE SAVE ERROR] {e}")
            # Continue anyway, caching is optional

    return jsonify({
        "status": "finished",
        "result": result,
        "cached": False,
    })


@bp.route("/flood-risk/<job_id>", methods=["GET"])
def get_flood_risk(job_id):
    """
    Poll job status.
    GET /api/flood-risk/<job_id>
    """
    conn = current_app.redis
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return jsonify({"error": "Job not found"}), 404

    status = job.get_status()
    if status == "finished":
        result = job.result
        return jsonify({"status": "finished", "result": result})
    elif status in ("queued", "started", "deferred"):
        return jsonify({"status": status})
    else:
        return jsonify({"status": "failed"})


@bp.route("/coral-bleaching", methods=["GET"])
def coral_bleaching():
    """
    Return a placeholder coral-bleaching prediction bundle for SE Asian reef sites.
    Shape contract:
      centroids: list of { lat, lon, name, cluster_id }
      weeks: list of ISO week strings (n_weeks)
      metrics: { key: { min, max, label, unit } }
      severity_labels: { "0": str, "1": str, "2": str }
      models: { lstm: { predicted_class, risk_score, tsa_dhw_last, tsa_dhw_max,
                        filled_sst_c, n_records,
                        prob_none, prob_moderate, prob_severe } }
      All per-centroid arrays have shape [n_centroids][n_weeks].
    """
    rng = random.Random(42)

    centroids = [
        {"lat": 8.72,   "lon": 126.06, "name": "Tubbataha Reef",      "cluster_id": 0},
        {"lat": -8.50,  "lon": 119.55, "name": "Komodo Reef",          "cluster_id": 1},
        {"lat": 4.19,   "lon": 114.50, "name": "Semporna Reef",         "cluster_id": 2},
        {"lat": 9.87,   "lon": 124.14, "name": "Bohol Sea Reef",        "cluster_id": 3},
        {"lat": -1.47,  "lon": 130.80, "name": "Raja Ampat Reef",       "cluster_id": 4},
        {"lat": 6.85,   "lon": 116.95, "name": "Sipadan Reef",          "cluster_id": 5},
        {"lat": 14.95,  "lon": 119.92, "name": "Hundred Islands Reef",  "cluster_id": 6},
        {"lat": -8.75,  "lon": 115.18, "name": "Bali Reef",             "cluster_id": 7},
        {"lat": 3.55,   "lon": 103.43, "name": "Tioman Island Reef",    "cluster_id": 8},
        {"lat": 11.57,  "lon": 103.15, "name": "Koh Tao Reef",          "cluster_id": 9},
        {"lat": 7.78,   "lon": 98.30,  "name": "Similan Islands Reef",  "cluster_id": 10},
        {"lat": -5.47,  "lon": 105.25, "name": "Krakatau Reef",         "cluster_id": 11},
    ]

    # Generate 8 recent weekly labels
    today = datetime.utcnow()
    weeks = []
    for i in range(7, -1, -1):
        d = today - timedelta(weeks=i)
        weeks.append(d.strftime("%Y-W%V"))

    n_c = len(centroids)
    n_w = len(weeks)

    def rand_series(lo, hi):
        return [[round(rng.uniform(lo, hi), 3) for _ in range(n_w)] for _ in range(n_c)]

    risk_score   = rand_series(0.05, 0.95)
    tsa_dhw_last = rand_series(0.0, 18.0)
    tsa_dhw_max  = [[max(tsa_dhw_last[c]) for _ in range(n_w)] for c in range(n_c)]
    filled_sst_c = rand_series(26.0, 32.5)
    n_records    = [[rng.randint(10, 120) for _ in range(n_w)] for _ in range(n_c)]

    # Derive probabilities from risk_score
    prob_none     = [[round(max(0.0, 1.0 - risk_score[c][w] * 1.5), 3) for w in range(n_w)] for c in range(n_c)]
    prob_severe   = [[round(max(0.0, risk_score[c][w] - 0.4), 3) for w in range(n_w)] for c in range(n_c)]
    prob_moderate = [[round(max(0.0, 1.0 - prob_none[c][w] - prob_severe[c][w]), 3) for w in range(n_w)] for c in range(n_c)]

    def classify(r):
        if r >= 0.65:
            return 2
        if r >= 0.35:
            return 1
        return 0

    predicted_class = [[classify(risk_score[c][w]) for w in range(n_w)] for c in range(n_c)]

    all_risk   = [v for row in risk_score   for v in row]
    all_dhw    = [v for row in tsa_dhw_last for v in row]
    all_sst    = [v for row in filled_sst_c for v in row]

    bundle = {
        "bbox": {
            "min_lat": min(c["lat"] for c in centroids) - 2,
            "max_lat": max(c["lat"] for c in centroids) + 2,
            "min_lon": min(c["lon"] for c in centroids) - 2,
            "max_lon": max(c["lon"] for c in centroids) + 2,
        },
        "weeks": weeks,
        "centroids": centroids,
        "severity_labels": {"0": "No Bleaching", "1": "Moderate Bleaching", "2": "Severe Bleaching"},
        "metrics": {
            "risk_score":   {"min": round(min(all_risk), 3), "max": round(max(all_risk), 3), "label": "Risk Score",  "unit": ""},
            "tsa_dhw_last": {"min": round(min(all_dhw), 1),  "max": round(max(all_dhw), 1),  "label": "DHW (last)",  "unit": "°C-wk"},
            "filled_sst_c": {"min": round(min(all_sst), 1),  "max": round(max(all_sst), 1),  "label": "SST",         "unit": "°C"},
        },
        "models": {
            "lstm": {
                "predicted_class": predicted_class,
                "risk_score":      risk_score,
                "tsa_dhw_last":    tsa_dhw_last,
                "tsa_dhw_max":     tsa_dhw_max,
                "filled_sst_c":    filled_sst_c,
                "n_records":       n_records,
                "prob_none":       prob_none,
                "prob_moderate":   prob_moderate,
                "prob_severe":     prob_severe,
            }
        },
    }
    return jsonify(bundle)


@bp.route("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

@bp.route("/config/maps-key")
def maps_key():
    """Provide the Google Maps API key to frontend."""
    key = current_app.config.get("GOOGLE_MAPS_API_KEY", "")
    return {"googleMapsApiKey": key}

@bp.route('/register', methods=['POST'])
@cross_origin(origins="https://aquatic-sustainability-834508815183.us-east1.run.app/", methods=["POST", "OPTIONS"])
def register():
    print("registering attempt now")
    if not supabase:
        return jsonify({"error": "Authentication service not configured"}), 503

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # create user
    raw = supabase.auth.sign_up({
        "email": email,
        "password": password
    })

    # Normalize response: supabase client may return different shapes depending on version.
    error = None
    user_data = None
    try:
        if isinstance(raw, dict):
            error = raw.get('error')
            user_data = raw.get('data')
        else:
            error = getattr(raw, 'error', None)
            user_data = getattr(raw, 'data', None)
    except Exception:
        error = None

    print({"raw": raw, "normalized_error": error, "normalized_data": user_data})

    if error:
        # error may be a dict or string
        if isinstance(error, dict) and error.get('message'):
            return jsonify({"error": error['message']}), 400
        return jsonify({"error": str(error)}), 400

    return jsonify({"message": "User registered successfully", "user": user_data}), 200


@bp.route('/login', methods=['POST'])
def login():
    if not supabase:
        return jsonify({"error": "Authentication service not configured"}), 503

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    raw = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    # Normalize response
    error = None
    data_obj = None
    try:
        if isinstance(raw, dict):
            error = raw.get('error')
            data_obj = raw.get('data')
        else:
            error = getattr(raw, 'error', None)
            data_obj = getattr(raw, 'data', None)
    except Exception:
        error = None

    if error:
        if isinstance(error, dict) and error.get('message'):
            return jsonify({"error": error['message']}), 400
        return jsonify({"error": str(error)}), 400

    # Try to extract session from data_obj (object or dict)
    session = None
    if hasattr(data_obj, 'session'):
        session = getattr(data_obj, 'session', None)
    elif isinstance(data_obj, dict):
        session = data_obj.get('session')
    elif hasattr(raw, 'session'):
        session = getattr(raw, 'session', None)

    if not session:
        return jsonify({
            "error": "No session returned from auth provider. This may mean the account is not verified or Supabase returned a different response shape.",
            "data_obj": str(data_obj)
        }), 500

    # Extract tokens from session (object or dict)
    access_token = getattr(session, 'access_token', None)
    refresh_token = getattr(session, 'refresh_token', None)
    if not access_token and isinstance(session, dict):
        access_token = session.get('access_token')
    if not refresh_token and isinstance(session, dict):
        refresh_token = session.get('refresh_token')

    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token
    }), 200

@bp.route("/drought")
def get_drought_data():
    # Drought regions with detailed polygon coordinates for accurate shapes
    drought_regions = [
        {
            "name": "Central California",
            "coords": [
                {"lat": 36.7, "lng": -120.9},
                {"lat": 36.9, "lng": -120.5},
                {"lat": 37.1, "lng": -120.0},
                {"lat": 37.0, "lng": -119.4},
                {"lat": 36.5, "lng": -118.8},
                {"lat": 36.0, "lng": -118.9},
                {"lat": 35.6, "lng": -119.5},
                {"lat": 35.4, "lng": -120.2},
                {"lat": 35.7, "lng": -120.7},
            ],
            "risk": 0.82
        },
        {
            "name": "Texas Panhandle",
            "coords": [
                {"lat": 35.0, "lng": -102.0},
                {"lat": 35.4, "lng": -101.5},
                {"lat": 35.7, "lng": -101.0},
                {"lat": 35.8, "lng": -100.5},
                {"lat": 35.5, "lng": -100.2},
                {"lat": 34.9, "lng": -100.3},
                {"lat": 34.5, "lng": -100.7},
                {"lat": 34.2, "lng": -101.3},
                {"lat": 34.4, "lng": -101.8},
            ],
            "risk": 0.75
        },
        {
            "name": "Colorado Plains",
            "coords": [
                {"lat": 39.5, "lng": -105.5},
                {"lat": 39.9, "lng": -105.0},
                {"lat": 40.2, "lng": -104.5},
                {"lat": 40.3, "lng": -104.0},
                {"lat": 40.0, "lng": -103.6},
                {"lat": 39.5, "lng": -103.5},
                {"lat": 39.0, "lng": -103.8},
                {"lat": 38.7, "lng": -104.3},
                {"lat": 38.8, "lng": -104.9},
                {"lat": 39.1, "lng": -105.3},
            ],
            "risk": 0.68
        },
        {
            "name": "South Florida",
            "coords": [
                {"lat": 26.2, "lng": -81.9},
                {"lat": 26.5, "lng": -81.6},
                {"lat": 26.8, "lng": -81.3},
                {"lat": 27.0, "lng": -81.0},
                {"lat": 26.9, "lng": -80.7},
                {"lat": 26.5, "lng": -80.6},
                {"lat": 26.0, "lng": -80.7},
                {"lat": 25.7, "lng": -81.0},
                {"lat": 25.5, "lng": -81.4},
                {"lat": 25.7, "lng": -81.7},
            ],
            "risk": 0.45
        },
        {
            "name": "Arizona Desert Basin",
            "coords": [
                {"lat": 33.5, "lng": -112.5},
                {"lat": 33.9, "lng": -112.0},
                {"lat": 34.2, "lng": -111.5},
                {"lat": 34.3, "lng": -111.0},
                {"lat": 34.1, "lng": -110.6},
                {"lat": 33.7, "lng": -110.5},
                {"lat": 33.3, "lng": -110.8},
                {"lat": 33.1, "lng": -111.3},
                {"lat": 33.2, "lng": -111.9},
            ],
            "risk": 0.88
        },
        {
            "name": "Oklahoma Grasslands",
            "coords": [
                {"lat": 36.5, "lng": -98.5},
                {"lat": 36.9, "lng": -98.0},
                {"lat": 37.2, "lng": -97.5},
                {"lat": 37.3, "lng": -97.0},
                {"lat": 37.0, "lng": -96.6},
                {"lat": 36.6, "lng": -96.5},
                {"lat": 36.2, "lng": -96.8},
                {"lat": 36.0, "lng": -97.3},
                {"lat": 36.1, "lng": -97.9},
            ],
            "risk": 0.71
        },
        {
            "name": "Nevada High Desert",
            "coords": [
                {"lat": 39.5, "lng": -118.5},
                {"lat": 40.0, "lng": -118.0},
                {"lat": 40.4, "lng": -117.5},
                {"lat": 40.6, "lng": -117.0},
                {"lat": 40.4, "lng": -116.5},
                {"lat": 39.9, "lng": -116.3},
                {"lat": 39.4, "lng": -116.5},
                {"lat": 39.1, "lng": -117.0},
                {"lat": 39.2, "lng": -117.7},
                {"lat": 39.4, "lng": -118.2},
            ],
            "risk": 0.79
        },
        {
            "name": "New Mexico Plateau",
            "coords": [
                {"lat": 35.0, "lng": -107.5},
                {"lat": 35.4, "lng": -107.0},
                {"lat": 35.7, "lng": -106.5},
                {"lat": 35.8, "lng": -106.0},
                {"lat": 35.5, "lng": -105.6},
                {"lat": 35.1, "lng": -105.5},
                {"lat": 34.7, "lng": -105.8},
                {"lat": 34.5, "lng": -106.3},
                {"lat": 34.6, "lng": -106.9},
            ],
            "risk": 0.73
        }
    ]

    return jsonify(drought_regions)

@bp.route("/flood")
def get_flood_data():
    # Flood zones with detailed polygon coordinates for accurate shapes
    flood_zones = [
        {
            "name": "Houston Metro Area",
            "coordinates": [
                {"lat": 29.7604, "lng": -95.3698},
                {"lat": 29.8504, "lng": -95.3298},
                {"lat": 29.9104, "lng": -95.4198},
                {"lat": 29.9404, "lng": -95.5298},
                {"lat": 29.8804, "lng": -95.6498},
                {"lat": 29.7804, "lng": -95.6898},
                {"lat": 29.6704, "lng": -95.5898},
                {"lat": 29.6404, "lng": -95.4598},
                {"lat": 29.6904, "lng": -95.3598},
            ],
            "probability": 0.78,
            "depth": "3.2 - 4.8 meters",
            "soilMoisture": "92%",
            "velocity": "2.3 m/s",
            "duration": "48-72 hours",
            "precipitation": "254 mm/24h",
            "elevation": "15-25m above sea level",
            "populationRisk": "~425,000",
            "infrastructureImpact": "Critical - highways, power grid",
            "lastIncident": "2024-02-15",
            "severity": "high"
        },
        {
            "name": "Mississippi River Delta",
            "coordinates": [
                {"lat": 29.9546, "lng": -90.0751},
                {"lat": 30.0846, "lng": -90.0251},
                {"lat": 30.1746, "lng": -90.1151},
                {"lat": 30.2146, "lng": -90.2451},
                {"lat": 30.1546, "lng": -90.3851},
                {"lat": 30.0346, "lng": -90.4251},
                {"lat": 29.9146, "lng": -90.3451},
                {"lat": 29.8746, "lng": -90.2051},
            ],
            "probability": 0.85,
            "depth": "4.5 - 6.2 meters",
            "soilMoisture": "95%",
            "velocity": "3.1 m/s",
            "duration": "72-96 hours",
            "precipitation": "305 mm/24h",
            "elevation": "5-15m above sea level",
            "populationRisk": "~180,000",
            "infrastructureImpact": "Severe - ports, refineries",
            "lastIncident": "2024-03-02",
            "severity": "high"
        },
        {
            "name": "Sacramento Valley",
            "coordinates": [
                {"lat": 38.5816, "lng": -121.4944},
                {"lat": 38.6816, "lng": -121.4344},
                {"lat": 38.7516, "lng": -121.5244},
                {"lat": 38.7916, "lng": -121.6544},
                {"lat": 38.7216, "lng": -121.7844},
                {"lat": 38.6116, "lng": -121.8244},
                {"lat": 38.5016, "lng": -121.7344},
                {"lat": 38.4716, "lng": -121.5944},
            ],
            "probability": 0.62,
            "depth": "2.1 - 3.5 meters",
            "soilMoisture": "88%",
            "velocity": "1.8 m/s",
            "duration": "36-48 hours",
            "precipitation": "178 mm/24h",
            "elevation": "8-18m above sea level",
            "populationRisk": "~310,000",
            "infrastructureImpact": "Moderate - residential, agriculture",
            "lastIncident": "2024-01-20",
            "severity": "medium"
        },
        {
            "name": "Red River Valley",
            "coordinates": [
                {"lat": 46.8772, "lng": -96.7898},
                {"lat": 46.9672, "lng": -96.7298},
                {"lat": 47.0472, "lng": -96.8098},
                {"lat": 47.0872, "lng": -96.9398},
                {"lat": 47.0272, "lng": -97.0698},
                {"lat": 46.9172, "lng": -97.1098},
                {"lat": 46.8072, "lng": -97.0298},
                {"lat": 46.7772, "lng": -96.8898},
            ],
            "probability": 0.71,
            "depth": "2.8 - 4.2 meters",
            "soilMoisture": "90%",
            "velocity": "2.0 m/s",
            "duration": "60-84 hours",
            "precipitation": "203 mm/24h",
            "elevation": "230-240m above sea level",
            "populationRisk": "~95,000",
            "infrastructureImpact": "Moderate - farms, roads",
            "lastIncident": "2024-03-10",
            "severity": "medium"
        },
        {
            "name": "Charleston Coastal Zone",
            "coordinates": [
                {"lat": 32.7765, "lng": -79.9311},
                {"lat": 32.8565, "lng": -79.8711},
                {"lat": 32.9165, "lng": -79.9411},
                {"lat": 32.9365, "lng": -80.0611},
                {"lat": 32.8765, "lng": -80.1811},
                {"lat": 32.7765, "lng": -80.2111},
                {"lat": 32.6865, "lng": -80.1411},
                {"lat": 32.6665, "lng": -80.0211},
            ],
            "probability": 0.56,
            "depth": "1.5 - 2.8 meters",
            "soilMoisture": "85%",
            "velocity": "1.4 m/s",
            "duration": "24-36 hours",
            "precipitation": "152 mm/24h",
            "elevation": "3-10m above sea level",
            "populationRisk": "~145,000",
            "infrastructureImpact": "Low - historic district, tourism",
            "lastIncident": "2024-02-28",
            "severity": "low"
        },
        {
            "name": "Cedar Rapids Region",
            "coordinates": [
                {"lat": 42.0083, "lng": -91.6444},
                {"lat": 42.0883, "lng": -91.5844},
                {"lat": 42.1483, "lng": -91.6544},
                {"lat": 42.1683, "lng": -91.7744},
                {"lat": 42.1083, "lng": -91.8944},
                {"lat": 42.0083, "lng": -91.9244},
                {"lat": 41.9283, "lng": -91.8544},
                {"lat": 41.9083, "lng": -91.7344},
            ],
            "probability": 0.68,
            "depth": "2.5 - 3.9 meters",
            "soilMoisture": "89%",
            "velocity": "1.9 m/s",
            "duration": "48-60 hours",
            "precipitation": "190 mm/24h",
            "elevation": "220-230m above sea level",
            "populationRisk": "~130,000",
            "infrastructureImpact": "Moderate - industrial, commercial",
            "lastIncident": "2024-03-05",
            "severity": "medium"
        }
    ]

    return jsonify(flood_zones)
