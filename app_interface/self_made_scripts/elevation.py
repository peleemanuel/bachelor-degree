#!/usr/bin/env python3

import os
import sys
import time
import requests
from dotenv import load_dotenv
from tinydb import TinyDB, Query


load_dotenv()  
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not set in .env")
    sys.exit(1)

DB_PATH = "elevation_db.json"

def fetch_elevation(lat: float, lon: float) -> dict:
    """
    Fetch elevation data for a single lat/lon using the Google Elevation API.
    """
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {"locations": f"{lat},{lon}", "key": API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()  # Elevation API returns JSON :contentReference[oaicite:5]{index=5}
        status = data.get("status")

        if status == "OK":
            return data
        else:
            raise Exception(f"Elevation API error: {status}; response: {data}")
    except (requests.exceptions.RequestException, ValueError) as e:
        raise Exception(f"Failed to fetch elevation for ({lat}, {lon}): {e}")

def save_to_db(lat: float, lon: float, elevation_data: dict):
    """
    Save the elevation result to TinyDB, keyed by lat/lon to avoid duplicates.
    """
    # Open (or create) the TinyDB JSON file
    db = TinyDB(DB_PATH)
    Location = Query()

    # Check again inside save_to_db to be safe
    if db.search((Location.lat == lat) & (Location.lon == lon)):
        print(f"[SAVE] ({lat}, {lon}) already exists in DB; skipping insert.")
        return

    # Use the first result (single-location query one element in results[])
    result = elevation_data["results"][0]
    entry = {
        "lat": lat,
        "lon": lon,
        "elevation": result.get("elevation"),   
        "resolution": result.get("resolution"), 
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    db.insert(entry)
    print(f"[DB] Inserted new record: {entry}")

def process_coordinates(coordinates):
    list_of_altitudes = []
    # Initialize TinyDB
    db = TinyDB(DB_PATH)
    Location = Query()

    for lat, lon in coordinates:
        # Check if an entry with this (lat, lon) already exists
        existing = db.search((Location.lat == lat) & (Location.lon == lon))
        if existing:
            list_of_altitudes.append(existing[0]["elevation"])
            print(f"[SKIP] ({lat}, {lon}) already in DB.")
            continue

        # If not, fetch elevation from the API
        try:
            data = fetch_elevation(lat, lon)
        except Exception as e:
            print(f"[ERROR] Could not fetch elevation for ({lat}, {lon}): {e}")
            continue

        # Save the new data to TinyDB
        save_to_db(lat, lon, data)
        list_of_altitudes.append(data["results"][0]["elevation"])

    print(f"[MAIN] Completed. Database stored at '{DB_PATH}'. Total records: {len(db)}")
    return list_of_altitudes

if __name__ == "__main__":
    # Example coordinates; replace with your source
    coordinates = [
        (46.2865373, 21.3863903),
        (40.748817, -73.985428),
        (37.774929, -122.419416)
    ]
    process_coordinates(coordinates)


