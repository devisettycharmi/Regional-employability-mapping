from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim

from .config import GEO_DIR


MANUAL_GEO_OVERRIDES = {
    "Warangal": (17.9893, 79.5941),
    "Hanamkonda": (17.9893, 79.5941),
    "Narsampet": (17.9812, 79.9736),
}


def ensure_cache() -> Path:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = GEO_DIR / "geocode_cache.csv"
    if not cache_path.exists():
        with cache_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["query", "latitude", "longitude", "formatted_address"])
    return cache_path


def geocode_location(query: str, user_agent: str = "employability-mapping") -> tuple[float | None, float | None]:
    cache_path = ensure_cache()
    df = pd.read_csv(cache_path)
    match = df[df["query"].astype(str).str.lower() == query.lower()]
    if not match.empty:
        return float(match.iloc[0]["latitude"]), float(match.iloc[0]["longitude"])
    if query in MANUAL_GEO_OVERRIDES:
        lat, lon = MANUAL_GEO_OVERRIDES[query]
        with cache_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([query, lat, lon, query])
        return lat, lon
    geolocator = Nominatim(user_agent=user_agent)
    location = geolocator.geocode(query, timeout=10)
    time.sleep(1.1)
    if location is None:
        return None, None
    lat, lon = location.latitude, location.longitude
    with cache_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([query, lat, lon, location.address])
    return lat, lon


def run_geocoding(students: pd.DataFrame) -> pd.DataFrame:
    locations = pd.concat([
        students["college"].dropna().astype(str),
        students["college_locality"].dropna().astype(str),
        students["home_locality"].dropna().astype(str),
    ]).drop_duplicates().reset_index(drop=True)
    results = []
    for value in locations:
        lat, lon = geocode_location(f"{value}, Warangal, Telangana, India")
        results.append({"location": value, "latitude": lat, "longitude": lon})
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Geocoding helpers ready.")
