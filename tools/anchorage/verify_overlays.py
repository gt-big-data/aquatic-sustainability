"""Validate the shared Anchorage GeoJSON export contract and model manifest."""

import argparse
import copy
import json
import math
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
DATE_FIELDS = {"from_survey", "predicted_for"}
VALUE_FIELDS = {"depth_ft", "change_ft"}


class ContractError(ValueError):
    """An output differs from the agreed forecast interchange format."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    def reject_constant(value):
        raise ContractError(f"{path} contains the nonfinite JSON constant {value}.")

    with path.open() as stream:
        return json.load(stream, parse_constant=reject_constant)


def forecast_dates(properties, context):
    require(isinstance(properties, dict) and set(properties) == DATE_FIELDS,
            f"{context} must have exactly from_survey and predicted_for properties.")
    parsed = []
    for key in ("from_survey", "predicted_for"):
        value = properties[key]
        require(isinstance(value, str), f"{context}.{key} must be an ISO date string.")
        try:
            day = date.fromisoformat(value)
        except ValueError as error:
            raise ContractError(f"{context}.{key} is an invalid date.") from error
        require(day.isoformat() == value, f"{context}.{key} must use YYYY-MM-DD.")
        parsed.append(day)
    require((parsed[1] - parsed[0]).days == 28, f"{context} must describe a 28-day forecast.")
    return tuple(parsed)


def polygon_key(geometry, context):
    require(isinstance(geometry, dict) and geometry.get("type") == "Polygon",
            f"{context} must be a Polygon.")
    coordinates = geometry.get("coordinates")
    require(isinstance(coordinates, list) and len(coordinates) == 1,
            f"{context} must have one exterior ring.")
    ring = coordinates[0]
    require(isinstance(ring, list) and len(ring) == 5,
            f"{context} must contain four corners and the repeated first corner.")
    for point in ring:
        require(isinstance(point, list) and len(point) == 2 and all(finite_number(v) for v in point),
                f"{context} coordinates must be finite longitude/latitude pairs.")
        require(-180 <= point[0] <= 180 and -90 <= point[1] <= 90,
                f"{context} longitude/latitude coordinates are out of range.")
    require(ring[0] == ring[-1], f"{context} ring is not closed.")
    require(len({tuple(point) for point in ring[:-1]}) == 4,
            f"{context} must contain four distinct corners.")
    # Subtract an origin before calculating area to avoid cancellation at small cells.
    points = [(p[0] - ring[0][0], p[1] - ring[0][1]) for p in ring]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:]))
    require(abs(area) > 1e-18, f"{context} polygon is degenerate or self-crossing.")
    return tuple(sorted((round(p[0], 9), round(p[1], 9)) for p in ring[:-1]))


def validate_collection(collection, context="GeoJSON", expected_dates=None):
    require(isinstance(collection, dict) and collection.get("type") == "FeatureCollection",
            f"{context} must be a FeatureCollection.")
    dates = forecast_dates(collection.get("properties"), context)
    if expected_dates is not None:
        require(dates == expected_dates, f"{context} dates differ from its manifest frame.")
    features = collection.get("features")
    require(isinstance(features, list) and len(features) > 0, f"{context} must contain predicted cells.")
    polygons = set()
    for index, feature in enumerate(features):
        name = f"{context}, feature {index}"
        require(isinstance(feature, dict) and feature.get("type") == "Feature",
                f"{name} must be a Feature.")
        properties = feature.get("properties")
        require(isinstance(properties, dict) and set(properties) == VALUE_FIELDS,
                f"{name} must have exactly depth_ft and change_ft properties.")
        require(all(finite_number(value) for value in properties.values()),
                f"{name} prediction properties must be finite numbers.")
        key = polygon_key(feature.get("geometry"), name)
        require(key not in polygons, f"{name} repeats an already exported polygon.")
        polygons.add(key)
    return polygons


def resolve_asset(url, static_root):
    require(isinstance(url, str), "An output URL must be a string.")
    parts = urlsplit(url)
    require(not any((parts.scheme, parts.netloc, parts.query, parts.fragment)),
            f"Output URL must be a local static asset: {url}")
    path = unquote(parts.path)
    require(path.startswith("/static/"), f"Output URL must start with /static/: {url}")
    destination = (static_root / path.removeprefix("/static/")).resolve()
    require(destination.is_relative_to(static_root.resolve()), f"Output URL escapes the static directory: {url}")
    require(destination.suffix == ".geojson" and destination.is_file(), f"Output GeoJSON is missing: {url}")
    return destination


def verify(manifest_path, static_root, reference_path=None):
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
            "Expected manifest schema_version 1.")
    models = manifest.get("models")
    require(isinstance(models, list) and len(models) > 0, "Manifest must list at least one model.")
    identifiers = set()
    for model in models:
        require(isinstance(model, dict), "Each model must be an object.")
        identifier = model.get("id")
        require(isinstance(identifier, str) and bool(identifier) and identifier not in identifiers,
                "Model IDs must be unique nonempty strings.")
        require(isinstance(model.get("label"), str) and bool(model["label"]),
                f"Model {identifier} needs a label.")
        require(finite_number(model.get("mae_ft")) and model["mae_ft"] >= 0,
                f"Model {identifier} needs a finite, nonnegative mae_ft.")
        identifiers.add(identifier)

    reference_polygons = None
    if reference_path is not None:
        reference_polygons = validate_collection(read_json(reference_path), "Reference attachment")
    frames = manifest.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, "Manifest must list forecast frames.")
    frame_ids, date_pairs = set(), set()
    files_checked, cells_checked = 0, 0
    for frame in frames:
        require(isinstance(frame, dict), "Each frame must be an object.")
        identifier = frame.get("id")
        require(isinstance(identifier, str) and bool(identifier) and identifier not in frame_ids,
                "Frame IDs must be unique nonempty strings.")
        frame_ids.add(identifier)
        dates = forecast_dates({key: frame.get(key) for key in DATE_FIELDS}, f"Frame {identifier}")
        require(dates not in date_pairs, "Manifest repeats a forecast date pair.")
        date_pairs.add(dates)
        outputs = frame.get("outputs")
        require(isinstance(outputs, dict) and set(outputs) == identifiers,
                f"Frame {identifier} must provide exactly one output for every registered model.")
        shared_polygons = None
        for model_id, url in outputs.items():
            path = resolve_asset(url, static_root)
            polygons = validate_collection(read_json(path), str(path), dates)
            if shared_polygons is None:
                shared_polygons = polygons
            else:
                require(polygons == shared_polygons,
                        f"Frame {identifier} model {model_id} differs from the shared cell footprint.")
            if reference_polygons is not None:
                require(polygons <= reference_polygons,
                        f"Frame {identifier} model {model_id} exports cells outside the reference footprint.")
            files_checked += 1
            cells_checked += len(polygons)
    return {"models": len(models), "frames": len(frames), "geojson_files_checked": files_checked,
            "predicted_cells_checked": cells_checked, "shared_footprints_verified": True,
            "reference_footprint_verified": reference_path is not None}


def self_test():
    valid = {
        "type": "FeatureCollection",
        "properties": {"from_survey": "2026-09-17", "predicted_for": "2026-10-15"},
        "features": [{"type": "Feature", "properties": {"depth_ft": 40.0, "change_ft": -0.5},
                      "geometry": {"type": "Polygon", "coordinates": [[
                          [-149.9, 61.2], [-149.8998, 61.2], [-149.8998, 61.2001],
                          [-149.9, 61.2001], [-149.9, 61.2],
                      ]]}}],
    }
    validate_collection(valid)
    invalid_cases = []
    missing_value = copy.deepcopy(valid)
    del missing_value["features"][0]["properties"]["depth_ft"]
    invalid_cases.append(missing_value)
    nonfinite = copy.deepcopy(valid)
    nonfinite["features"][0]["properties"]["change_ft"] = math.nan
    invalid_cases.append(nonfinite)
    unclosed = copy.deepcopy(valid)
    unclosed["features"][0]["geometry"]["coordinates"][0][-1] = [-149.9, 61.2002]
    invalid_cases.append(unclosed)
    wrong_dates = copy.deepcopy(valid)
    wrong_dates["properties"]["predicted_for"] = "2026-10-16"
    invalid_cases.append(wrong_dates)
    duplicate = copy.deepcopy(valid)
    duplicate["features"].append(copy.deepcopy(duplicate["features"][0]))
    invalid_cases.append(duplicate)
    for invalid in invalid_cases:
        try:
            validate_collection(invalid)
        except ContractError:
            continue
        raise AssertionError("The validator accepted a malformed GeoJSON fixture.")
    return {"valid_fixture_accepted": True, "malformed_fixtures_rejected": len(invalid_cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path,
                        default=ROOT / "backend/static/data/anchorage/manifest.json")
    parser.add_argument("--static-root", type=Path, default=ROOT / "backend/static")
    parser.add_argument("--reference", type=Path, help="Attachment used to check polygon footprint containment.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        result = self_test() if args.self_test else verify(args.manifest, args.static_root, args.reference)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Overlay contract check failed. {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
