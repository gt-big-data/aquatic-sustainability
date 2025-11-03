from flask import Blueprint, current_app
from flask import Flask, jsonify
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