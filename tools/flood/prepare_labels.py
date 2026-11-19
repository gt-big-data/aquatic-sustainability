"""Audit Dean-format labels and prepare reviewable chronological partitions.

Uses only the standard library. Does not download data or create negative labels.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


def distance_km(first, second):
    lat1, lon1, lat2, lon2 = map(math.radians, (*first, *second))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1, max(0, value))))


def prepare(rows, validation_start, test_start, radius_km=100, buffer_days=14, require_smap=False):
    if validation_start >= test_start:
        raise ValueError("Validation must start before testing.")
    samples, rejected, seen = [], [], set()
    for line, original in enumerate(rows, 2):
        row = dict(original)
        row.update(source_line=line, issue="")
        try:
            when = date.fromisoformat(str(original["Began"]).strip()[:10])
            end = date.fromisoformat(str(original.get("Ended") or when).strip()[:10])
            lat, lon = float(original["lat"]), float(original["long"])
            label_text = str(original["isFlood"]).strip().lower()
            label = {"0": 0, "0.0": 0, "false": 0, "1": 1, "1.0": 1, "true": 1}[label_text]
            if not all(map(math.isfinite, (lat, lon))) or not (-90 <= lat <= 90 and -180 <= lon <= 180) or end < when:
                raise ValueError("Invalid coordinates or dates")
            if not str(original.get("ID", "")).strip() or not str(original.get("Country", "")).strip():
                raise ValueError("ID and Country are required")
        except (KeyError, TypeError, ValueError):
            row["issue"] = "invalid_or_missing_required_field"
            rejected.append(row)
            continue
        row.update(Began=when.isoformat(), Ended=end.isoformat(), lat=lat, long=lon, isFlood=label)
        key = (round(lat, 6), round(lon, 6), when, end, label)
        if key in seen:
            row["issue"] = "duplicate_location_dates_label"
            rejected.append(row)
            continue
        seen.add(key)
        if require_smap and when - timedelta(days=22) < date(2015, 3, 31):
            row["issue"] = "required_smap_history_predates_product"
            rejected.append(row)
            continue
        row["split"] = "test" if when >= test_start else "validation" if when >= validation_start else "train"
        row["_start"], row["_end"] = when, end
        row["_group"] = str(original.get("event_group") or original["ID"]).strip()
        samples.append(row)

    positives = [row for row in samples if row["isFlood"] == 1]
    groups = defaultdict(set)
    for row in samples:
        groups[row["_group"]].add(row["split"])
    crossing_groups = {group for group, splits in groups.items() if len(splits) > 1}
    accepted = []
    for row in samples:
        issues = []
        if row["_group"] in crossing_groups:
            issues.append("event_group_crosses_split_boundary")
        boundary = validation_start if row["split"] == "train" else test_start if row["split"] == "validation" else None
        if boundary and row["_end"] >= boundary:
            issues.append("event_duration_crosses_split_boundary")
        if row["isFlood"] == 0:
            for flood in positives:
                overlapping = row["_start"] <= flood["_end"] + timedelta(days=buffer_days) and row["_end"] >= flood["_start"] - timedelta(days=buffer_days)
                if overlapping and distance_km((row["lat"], row["long"]), (flood["lat"], flood["long"])) <= radius_km:
                    issues.append("negative_near_recorded_flood_needs_review")
                    break
            if str(row.get("negative_verified", "")).strip().lower() not in {"1", "true", "yes"}:
                issues.append("negative_not_independently_verified")
        if issues:
            row["issue"] = ";".join(issues)
            rejected.append(row)
        else:
            accepted.append(row)

    # Proximity is a warning, not a claim that two records describe one storm.
    cross_split_pairs = 0
    for index, first in enumerate(positives):
        for second in positives[index + 1:]:
            if first["split"] != second["split"] and first["_start"] <= second["_end"] + timedelta(days=buffer_days) and second["_start"] <= first["_end"] + timedelta(days=buffer_days):
                if distance_km((first["lat"], first["long"]), (second["lat"], second["long"])) <= radius_km:
                    cross_split_pairs += 1
    counts = Counter(f"{row['split']}:{row['isFlood']}" for row in accepted)
    report = {
        "accepted_rows": len(accepted), "quarantined_rows": len(rejected),
        "accepted_split_label_counts": dict(counts),
        "issue_counts": dict(Counter(issue for row in rejected for issue in row["issue"].split(";"))),
        "nearby_cross_split_positive_pairs_for_review": cross_split_pairs,
        "both_classes_in_every_split": all(counts[f"{split}:{label}"] > 0 for split in ("train", "validation", "test") for label in (0, 1)),
        "settings": {"validation_start": str(validation_start), "test_start": str(test_start), "radius_km": radius_km, "buffer_days": buffer_days, "require_smap": require_smap},
        "limitations": [
            "No independent label validation is performed; negative_verified must come from documented review.",
            "Radius/date screening cannot rule out undocumented floods or shared storms outside the buffer.",
            "Provide event_group for records belonging to the same storm, including paired controls.",
            "Satellite publication times and input-window split boundaries must be checked when assembling features.",
            "Do not interpret accepted rows as a certified training dataset.",
        ],
    }
    def clean(row):
        return {key: value for key, value in row.items() if not key.startswith("_")}
    return list(map(clean, accepted)), list(map(clean, rejected)), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--validation-start", type=date.fromisoformat, required=True)
    parser.add_argument("--test-start", type=date.fromisoformat, required=True)
    parser.add_argument("--radius-km", type=float, default=100)
    parser.add_argument("--buffer-days", type=int, default=14)
    parser.add_argument("--require-smap", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / ".local")
    args = parser.parse_args()
    if args.radius_km <= 0 or args.buffer_days < 0:
        parser.error("radius must be positive and buffer days nonnegative")
    with args.csv.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    accepted, rejected, report = prepare(rows, args.validation_start, args.test_start, args.radius_km, args.buffer_days, args.require_smap)
    report["source_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, output_rows in [("accepted.csv", accepted), ("quarantine.csv", rejected)]:
        fields = sorted({key for row in output_rows for key in row}) or ["ID", "issue"]
        with (args.output / filename).open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
