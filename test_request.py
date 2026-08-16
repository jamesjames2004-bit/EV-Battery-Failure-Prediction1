"""Example call to the running API (local or deployed)."""
import requests

url = "http://localhost:8000/predict"  # swap for your deployed URL

payload = {
    "vehicle_id": "EV118422",
    "readings": {
        "battery_health_percent": 68.5,
        "state_of_health": 69.1,
        "capacity_loss_percent": 22.4,
        "cell_voltage_std": 0.041,
        "cycle_count": 1420,
        "internal_resistance": 1.8,
        "thermal_runaway_risk": 62.0,
        "vehicle_brand": "Tesla",
        "battery_chemistry": "NMC",
        "fleet_or_private": "Fleet"
        # ...any other raw fields available; missing ones fall back to training medians
    }
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
