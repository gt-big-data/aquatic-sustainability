# app/tasks.py

from .flood_model import run_flood_inference

def run_flood_job(center_lat, center_lon):
    """
    Thin wrapper so RQ can call this function.
    """
    return run_flood_inference(center_lat, center_lon)
