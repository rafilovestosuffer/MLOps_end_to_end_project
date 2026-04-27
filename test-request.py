"""
Demo API test — sends all three profiles to the running Flask server.
Make sure `python app.py` is running before executing this script.
"""
import requests, json

BASE_URL = "http://localhost:5000"

PROFILES = [
    {
        "name": "Corporate Executive (expected >50K)",
        "payload": {
            "age": 45, "workclass": 3, "education_num": 14,
            "marital_status": 2, "occupation": 3, "relationship": 0,
            "race": 4, "sex": 1, "capital_gain": 15024,
            "capital_loss": 0, "hours_per_week": 50, "native_country": 38
        }
    },
    {
        "name": "Government Clerk (expected <=50K)",
        "payload": {
            "age": 39, "workclass": 6, "education_num": 13,
            "marital_status": 4, "occupation": 0, "relationship": 1,
            "race": 4, "sex": 1, "capital_gain": 2174,
            "capital_loss": 0, "hours_per_week": 40, "native_country": 38
        }
    },
    {
        "name": "Early Career Worker (expected <=50K)",
        "payload": {
            "age": 24, "workclass": 3, "education_num": 10,
            "marital_status": 4, "occupation": 7, "relationship": 3,
            "race": 4, "sex": 0, "capital_gain": 0,
            "capital_loss": 0, "hours_per_week": 30, "native_country": 38
        }
    }
]


def check_health():
    r = requests.get(f"{BASE_URL}/health")
    print(f"Health: {r.json()}\n")


def run_predictions():
    for profile in PROFILES:
        print(f"─── {profile['name']} ───")
        r = requests.post(
            f"{BASE_URL}/predict",
            json=profile["payload"],
            headers={"Content-Type": "application/json"}
        )
        result = r.json()
        print(f"  Prediction : {result.get('income_category', 'N/A')}")
        print(f"  Confidence : {result.get('confidence', 'N/A')}%")
        print(f"  Status     : {result.get('status')}\n")


if __name__ == "__main__":
    check_health()
    run_predictions()
