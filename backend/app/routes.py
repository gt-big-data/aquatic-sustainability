from flask import Blueprint, current_app, request, jsonify
from flask_cors import cross_origin
from rq.job import Job

from . import supabase
from .tasks import run_flood_job
import random

bp = Blueprint("api", __name__)
# app/routes.py


@bp.route("/flood-risk", methods=["POST"])
def start_flood_risk():
    """
    Synchronous flood risk endpoint (no Redis/RQ).
    POST /api/flood-risk
    Body: { "lat": <float>, "lon": <float> }
    """
    data = request.get_json() or {}
    try:
        center_lat = float(data["lat"])
        center_lon = float(data["lon"])
    except (KeyError, ValueError):
        return jsonify({"error": "lat and lon are required floats"}), 400

    try:
        # Directly run the job (this calls your model)
        result = run_flood_job(center_lat, center_lon)
    except Exception as e:
        # Log full traceback to the Flask console
        current_app.logger.exception("Error running flood job")
        return jsonify({"error": "internal error running model"}), 500

    # Return result immediately, no job_id / polling
    return jsonify({
        "status": "finished",
        "result": result,
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


@bp.route("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

@bp.route("/config/maps-key")
def maps_key():
    """Provide the Google Maps API key to frontend."""
    key = current_app.config.get("GOOGLE_MAPS_API_KEY", "")
    return {"googleMapsApiKey": key}

@bp.route("/config/mongoDB-uri")
def mongoDB_uri():
    """Provide MongoDB connnection URI"""
    key = current_app.config.get("MONGODB_URI", "")
    return {"mongoDBUri": key}

@bp.route('/register', methods=['POST'])
@cross_origin(origins="https://aquatic-sustainability.vercel.app", methods=["POST", "OPTIONS"])
def register():
    print("[REGISTER] Registration attempt")
    if not supabase:
        print("[REGISTER] ERROR: Supabase client not configured")
        return jsonify({"error": "Authentication service not configured"}), 503

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    print(f"[REGISTER] Attempting signup for: {email}")

    try:
        # create user
        raw = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
    except Exception as e:
        print(f"[REGISTER] Exception during signup: {e}")
        return jsonify({"error": str(e)}), 500

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

    print(f"[REGISTER] Response - Error: {error}, Data: {user_data}")

    if error:
        # error may be a dict or string
        if isinstance(error, dict) and error.get('message'):
            print(f"[REGISTER] Error message: {error['message']}")
            return jsonify({"error": error['message']}), 400
        print(f"[REGISTER] Error: {str(error)}")
        return jsonify({"error": str(error)}), 400

    print("[REGISTER] SUCCESS")
    return jsonify({"message": "User registered successfully", "user": user_data}), 200


@bp.route('/login', methods=['POST'])
def login():
    print("[LOGIN] Login attempt")
    if not supabase:
        print("[LOGIN] ERROR: Supabase client not configured")
        return jsonify({"error": "Authentication service not configured"}), 503

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    print(f"[LOGIN] Attempting login for: {email}")

    try:
        raw = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
    except Exception as e:
        print(f"[LOGIN] Exception during login: {e}")
        return jsonify({"error": str(e)}), 500

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

    print(f"[LOGIN] Response - Error: {error}, Data: {data_obj}")

    if error:
        if isinstance(error, dict) and error.get('message'):
            print(f"[LOGIN] Error message: {error['message']}")
            return jsonify({"error": error['message']}), 400
        print(f"[LOGIN] Error: {str(error)}")
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
