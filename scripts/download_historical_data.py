"""
Historical Data Download Script

Downloads F1 historical data from public sources (FastF1, Ergast API).
Stores data in raw data directory for processing.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import requests
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ErgastAPI:
    """Interface to Ergast F1 API for historical data."""

    BASE_URL = "http://ergast.com/api/f1"

    def __init__(self):
        self.session = requests.Session()

    def get_seasons(self, start_year: int = 2018, end_year: int = 2024) -> List[int]:
        """Get list of seasons to download."""
        return list(range(start_year, end_year + 1))

    def get_race_schedule(self, year: int) -> Dict:
        """Get race schedule for a season."""
        url = f"{self.BASE_URL}/{year}.json"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_race_results(self, year: int, round_num: int) -> Dict:
        """Get race results for a specific race."""
        url = f"{self.BASE_URL}/{year}/{round_num}/results.json"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_qualifying_results(self, year: int, round_num: int) -> Dict:
        """Get qualifying results."""
        url = f"{self.BASE_URL}/{year}/{round_num}/qualifying.json"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_lap_times(self, year: int, round_num: int) -> Dict:
        """Get all lap times for a race."""
        url = f"{self.BASE_URL}/{year}/{round_num}/laps.json?limit=2000"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_pit_stops(self, year: int, round_num: int) -> Dict:
        """Get pit stop data for a race."""
        url = f"{self.BASE_URL}/{year}/{round_num}/pitstops.json"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def setup_data_directories():
    """Create necessary data directories."""
    base_path = Path(__file__).parent.parent / "data" / "raw" / "historical"
    base_path.mkdir(parents=True, exist_ok=True)

    subdirs = ["races", "lap_times", "pit_stops", "qualifying"]
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)

    return base_path


def download_season_data(api: ErgastAPI, year: int, base_path: Path):
    """Download all data for a season."""
    print(f"\nDownloading data for {year} season...")

    try:
        # Get race schedule
        schedule = api.get_race_schedule(year)
        races = schedule["MRData"]["RaceTable"]["Races"]
        print(f"  Found {len(races)} races")

        for race in races:
            round_num = int(race["round"])
            race_name = race["raceName"]
            print(f"  Round {round_num}: {race_name}")

            try:
                # Race results
                results = api.get_race_results(year, round_num)
                results_path = base_path / "races" / f"{year}_R{round_num:02d}_results.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

                # Qualifying results
                qualifying = api.get_qualifying_results(year, round_num)
                qual_path = base_path / "qualifying" / f"{year}_R{round_num:02d}_qualifying.json"
                with open(qual_path, "w") as f:
                    json.dump(qualifying, f, indent=2)

                # Lap times
                lap_times = api.get_lap_times(year, round_num)
                laps_path = base_path / "lap_times" / f"{year}_R{round_num:02d}_laps.json"
                with open(laps_path, "w") as f:
                    json.dump(lap_times, f, indent=2)

                # Pit stops
                pit_stops = api.get_pit_stops(year, round_num)
                pits_path = base_path / "pit_stops" / f"{year}_R{round_num:02d}_pitstops.json"
                with open(pits_path, "w") as f:
                    json.dump(pit_stops, f, indent=2)

                print(f"    ✓ Downloaded all data")

            except Exception as e:
                print(f"    ✗ Error downloading race data: {e}")
                continue

    except Exception as e:
        print(f"  ✗ Error downloading season: {e}")


def download_fastf1_data(year: int, base_path: Path):
    """
    Download data using FastF1 library (if available).

    Note: FastF1 provides more detailed telemetry but requires installation.
    """
    try:
        import fastf1

        print(f"\nDownloading FastF1 data for {year}...")
        fastf1.Cache.enable_cache(str(base_path / "fastf1_cache"))

        # Get season schedule
        schedule = fastf1.get_event_schedule(year)

        for event in schedule.itertuples():
            try:
                print(f"  Event: {event.EventName}")

                # Load race session
                session = fastf1.get_session(year, event.RoundNumber, "R")
                session.load()

                # Save session data
                session_path = base_path / "fastf1" / f"{year}_R{event.RoundNumber:02d}_session.pkl"
                session_path.parent.mkdir(exist_ok=True)

                # Export to JSON (simplified)
                session_data = {
                    "event": event.EventName,
                    "date": str(event.EventDate),
                    "laps": len(session.laps),
                }

                with open(session_path.with_suffix(".json"), "w") as f:
                    json.dump(session_data, f, indent=2)

                print(f"    ✓ Downloaded FastF1 data")

            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

    except ImportError:
        print("\nNote: FastF1 library not installed. Skipping FastF1 data download.")
        print("Install with: pip install fastf1")


def main():
    """Main download function."""
    print("=" * 60)
    print("F1 Historical Data Download")
    print("=" * 60)

    # Setup directories
    base_path = setup_data_directories()
    print(f"\n✓ Data directory: {base_path}")

    # Initialize API
    api = ErgastAPI()

    # Get seasons to download
    seasons = api.get_seasons(start_year=2018, end_year=2024)
    print(f"\n✓ Will download {len(seasons)} seasons: {seasons[0]}-{seasons[-1]}")

    # Download each season
    for year in seasons:
        download_season_data(api, year, base_path)

    # Try to download FastF1 data
    for year in [2023, 2024]:  # Most recent seasons
        download_fastf1_data(year, base_path)

    print("\n" + "=" * 60)
    print("Historical data download completed!")
    print("=" * 60)
    print(f"\nData stored in: {base_path}")


if __name__ == "__main__":
    main()
