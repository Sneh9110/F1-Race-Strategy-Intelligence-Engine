"""Load tests using Locust.

Run with:
    locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10
"""

from locust import HttpUser, task, between
import random


class F1StrategyAPIUser(HttpUser):
    """Simulated user for load testing F1 Strategy API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Login and get token."""
        response = self.client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"}
        )
        if response.status_code == 200:
            self.token = response.json()["data"]["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}
    
    @task(10)
    def predict_lap_time(self):
        """Predict lap time (most common operation - 40% of traffic)."""
        circuits = ["Monaco", "Silverstone", "Monza", "Spa", "Singapore"]
        compounds = ["SOFT", "MEDIUM", "HARD"]
        
        payload = {
            "circuit_name": random.choice(circuits),
            "driver": "Max Verstappen",
            "team": "Red Bull Racing",
            "tire_compound": random.choice(compounds),
            "tire_age": random.randint(1, 30),
            "fuel_load": random.uniform(50, 110),
            "track_temp": random.uniform(25, 45),
            "air_temp": random.uniform(20, 35),
            "weather_condition": "Dry"
        }
        
        with self.client.post(
            "/api/v1/predict/laptime",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(5)
    def predict_tire_degradation(self):
        """Predict tire degradation (20% of traffic)."""
        circuits = ["Monaco", "Silverstone", "Monza", "Spa"]
        compounds = ["SOFT", "MEDIUM", "HARD"]
        
        payload = {
            "circuit_name": random.choice(circuits),
            "tire_compound": random.choice(compounds),
            "laps": random.randint(10, 30),
            "track_temp": random.uniform(25, 45),
            "fuel_load": random.uniform(50, 110),
            "downforce_level": random.choice(["Low", "Medium", "High"])
        }
        
        self.client.post(
            "/api/v1/predict/degradation",
            json=payload,
            headers=self.headers
        )
    
    @task(3)
    def predict_safety_car(self):
        """Predict safety car probability (12% of traffic)."""
        circuits = ["Baku", "Monaco", "Singapore", "Saudi Arabia"]
        
        payload = {
            "circuit_name": random.choice(circuits),
            "lap": random.randint(1, 50),
            "total_laps": 50,
            "weather_condition": random.choice(["Dry", "Wet"]),
            "incidents_so_far": random.randint(0, 3)
        }
        
        self.client.post(
            "/api/v1/predict/safety-car",
            json=payload,
            headers=self.headers
        )
    
    @task(2)
    def simulate_strategy(self):
        """Run strategy simulation (8% of traffic)."""
        circuits = ["Monaco", "Silverstone", "Spa"]
        
        payload = {
            "circuit_name": random.choice(circuits),
            "total_laps": random.randint(44, 78),
            "starting_tire": random.choice(["SOFT", "MEDIUM"]),
            "fuel_load": random.uniform(100, 110),
            "weather_condition": "Dry",
            "pit_stops": [
                {"lap": random.randint(15, 25), "tire": "MEDIUM"},
                {"lap": random.randint(35, 45), "tire": "HARD"}
            ]
        }
        
        self.client.post(
            "/api/v1/simulate/strategy",
            json=payload,
            headers=self.headers
        )
    
    @task(3)
    def recommend_strategy(self):
        """Get strategy recommendation (12% of traffic)."""
        circuits = ["Monaco", "Silverstone", "Spa", "Monza"]
        
        payload = {
            "circuit_name": random.choice(circuits),
            "current_lap": random.randint(10, 40),
            "total_laps": 50,
            "current_position": random.randint(1, 10),
            "current_tire": random.choice(["SOFT", "MEDIUM", "HARD"]),
            "tire_age": random.randint(5, 25),
            "fuel_remaining": random.uniform(30, 80),
            "gap_to_leader": random.uniform(0, 20),
            "weather_condition": "Dry",
            "safety_car_deployed": False
        }
        
        self.client.post(
            "/api/v1/strategy/recommend",
            json=payload,
            headers=self.headers
        )
    
    @task(2)
    def health_check(self):
        """Check API health (8% of traffic)."""
        self.client.get("/api/v1/health")
    
    @task(1)
    def get_stats(self):
        """Get prediction statistics (4% of traffic)."""
        self.client.get("/api/v1/predict/stats")


class AdminUser(HttpUser):
    """Admin user for testing authenticated endpoints."""
    
    wait_time = between(5, 10)
    
    def on_start(self):
        """Login as admin."""
        response = self.client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"}
        )
        if response.status_code == 200:
            self.token = response.json()["data"]["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task
    def list_decision_modules(self):
        """List decision modules (requires auth)."""
        self.client.get(
            "/api/v1/strategy/modules",
            headers=self.headers
        )
