"""
Database Setup Script - Initialize PostgreSQL/TimescaleDB Schema

Creates all required tables, indexes, and hypertables for time-series data.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime


def get_connection_string():
    """Get database connection string from environment."""
    return os.getenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:5432/f1_strategy"
    )


def create_database():
    """Create database if it doesn't exist."""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            "postgresql://postgres:password@localhost:5432/postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'f1_strategy'"
        )
        exists = cursor.fetchone()

        if not exists:
            cursor.execute("CREATE DATABASE f1_strategy")
            print("✓ Created database: f1_strategy")
        else:
            print("✓ Database f1_strategy already exists")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)


def setup_schema(conn):
    """Create all database tables and indexes."""
    cursor = conn.cursor()

    # Enable TimescaleDB extension (if available)
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        print("✓ Enabled TimescaleDB extension")
    except Exception as e:
        print(f"Note: TimescaleDB not available: {e}")

    # Create timing_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timing_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            session_id VARCHAR(100) NOT NULL,
            driver_number INT NOT NULL,
            lap_number INT NOT NULL,
            lap_time FLOAT,
            sector_1_time FLOAT,
            sector_2_time FLOAT,
            sector_3_time FLOAT,
            tire_compound VARCHAR(20),
            tire_age INT,
            position INT,
            gap_to_leader FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create weather_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            session_id VARCHAR(100) NOT NULL,
            track_temp_celsius FLOAT,
            air_temp_celsius FLOAT,
            humidity_percent FLOAT,
            wind_speed_kmh FLOAT,
            rainfall_mm FLOAT,
            track_condition VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create telemetry_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            session_id VARCHAR(100) NOT NULL,
            driver_number INT NOT NULL,
            lap_number INT,
            speed_kmh FLOAT,
            throttle_percent FLOAT,
            brake_percent FLOAT,
            gear INT,
            rpm INT,
            tire_temp_fl FLOAT,
            tire_temp_fr FLOAT,
            fuel_remaining_kg FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create historical_races table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_races (
            id SERIAL PRIMARY KEY,
            race_id VARCHAR(100) UNIQUE NOT NULL,
            year INT NOT NULL,
            track_name VARCHAR(100) NOT NULL,
            race_date DATE NOT NULL,
            winner_driver_number INT,
            total_laps INT,
            race_duration_seconds FLOAT,
            weather_conditions VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create safety_car_events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS safety_car_events (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR(100) UNIQUE NOT NULL,
            session_id VARCHAR(100) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            start_lap INT NOT NULL,
            end_lap INT NOT NULL,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ NOT NULL,
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create model_predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            session_id VARCHAR(100) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50),
            driver_number INT,
            prediction_type VARCHAR(50),
            prediction_value JSONB,
            confidence_score FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create strategy_recommendations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_recommendations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            session_id VARCHAR(100) NOT NULL,
            driver_number INT NOT NULL,
            current_lap INT NOT NULL,
            recommendation_type VARCHAR(50) NOT NULL,
            recommendation_data JSONB,
            priority VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    print("✓ Created all tables")

    # Create indexes for fast querying
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_timing_session ON timing_data(session_id, driver_number, lap_number)",
        "CREATE INDEX IF NOT EXISTS idx_timing_timestamp ON timing_data(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_weather_session ON weather_data(session_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry_data(session_id, driver_number, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_session ON model_predictions(session_id, model_name, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_strategies_session ON strategy_recommendations(session_id, driver_number, current_lap)",
    ]

    for index_sql in indexes:
        cursor.execute(index_sql)

    print("✓ Created all indexes")

    # Convert tables to hypertables (TimescaleDB)
    try:
        hypertables = [
            "SELECT create_hypertable('timing_data', 'timestamp', if_not_exists => TRUE)",
            "SELECT create_hypertable('weather_data', 'timestamp', if_not_exists => TRUE)",
            "SELECT create_hypertable('telemetry_data', 'timestamp', if_not_exists => TRUE)",
            "SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE)",
            "SELECT create_hypertable('strategy_recommendations', 'timestamp', if_not_exists => TRUE)",
        ]

        for hypertable_sql in hypertables:
            cursor.execute(hypertable_sql)

        print("✓ Converted tables to hypertables (TimescaleDB)")
    except Exception as e:
        print(f"Note: Hypertable conversion skipped (TimescaleDB not available): {e}")

    conn.commit()
    cursor.close()


def main():
    """Main setup function."""
    print("=" * 60)
    print("F1 Strategy Engine - Database Setup")
    print("=" * 60)

    # Create database
    create_database()

    # Connect to f1_strategy database
    try:
        conn = psycopg2.connect(get_connection_string())
        print("✓ Connected to database")

        # Setup schema
        setup_schema(conn)

        conn.close()
        print("\n" + "=" * 60)
        print("Database setup completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
