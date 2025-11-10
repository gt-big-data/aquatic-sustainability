from flask import Blueprint, current_app, request, jsonify, Flask
from flask_cors import cross_origin
from . import supabase
import random

bp = Blueprint("api", __name__)

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
@cross_origin(origins="https://aquatic-sustainability.vercel.app", methods=["POST", "OPTIONS"])
def register():
    print("registering attempt now")
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
    # Example drought regions (could later come from a database or model)
    drought_regions = [
        {
            "name": "Central California",
            "base_coords": [
                (36.7, -120.9), (36.5, -119.4), (35.8, -118.8), (35.4, -120.5)
            ]
        },
        {
            "name": "Texas Panhandle",
            "base_coords": [
                (35.0, -102.0), (35.5, -101.0), (34.5, -100.5), (34.2, -101.8)
            ]
        },
        {
            "name": "Colorado Plains",
            "base_coords": [
                (39.5, -105.5), (40.0, -104.2), (39.3, -103.8), (38.8, -104.9)
            ]
        },
        {
            "name": "South Florida",
            "base_coords": [
                (26.2, -81.9), (26.7, -81.3), (25.8, -80.8), (25.5, -81.7)
            ]
        }
    ]

    # Generate dynamic "risk" values and small coordinate variation for realism
    drought_data = []
    for region in drought_regions:
        risk = round(random.uniform(0.2, 0.9), 2)
        coords = [{"lat": lat + random.uniform(-0.1, 0.1),
                   "lng": lng + random.uniform(-0.1, 0.1)}
                   for (lat, lng) in region["base_coords"]]
        drought_data.append({
            "name": region["name"],
            "risk": risk,
            "coords": coords
        })

    return jsonify(drought_data)
