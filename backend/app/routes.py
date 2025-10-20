from flask import Blueprint, current_app

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