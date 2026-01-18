"""
Test script for SalesHouses FastAPI backend
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()

        health_data = response.json()
        print("✅ Health check successful!")
        print(f"   Status: {health_data['status']}")
        print(f"   Model loaded: {health_data['model_loaded']}")
        print(f"   Cities available: {len(health_data['available_cities'])}")
        print(f"   Equipment features: {len(health_data['equipment_features'])}")

        return True
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    print("\n🔍 Testing prediction endpoint...")

    # Test data
    test_apartment = {
        "city": "Casablanca",
        "surface_area": 100,
        "nb_baths": 2,
        "total_rooms": 3,
        "equipment_list": ["Ascenseur", "Balcon", "Parking"]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_apartment,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        prediction_data = response.json()
        print("✅ Prediction successful!")
        print(f"   City: {prediction_data['city']}")
        print(f"   Surface: {prediction_data['surface_area']} m²")
        print(f"   Predicted Price: {prediction_data['predicted_price']:,.0f} DH")
        print(f"   Price/m²: {prediction_data['price_per_m2']:,.0f} DH/m²")
        print(f"   Confidence Interval: {prediction_data['confidence_interval']['lower']:,.0f} - {prediction_data['confidence_interval']['upper']:,.0f} DH")

        return True
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return False

def test_invalid_city():
    """Test prediction with invalid city"""
    print("\n🔍 Testing invalid city handling...")

    invalid_apartment = {
        "city": "InvalidCity",
        "surface_area": 100,
        "nb_baths": 2,
        "total_rooms": 3,
        "equipment_list": []
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_apartment,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 422:  # Validation error
            print("✅ Invalid city properly rejected!")
            print(f"   Error: {response.json()['detail']}")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Invalid city test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing SalesHouses FastAPI Backend")
    print("=" * 50)
    print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API URL: {BASE_URL}")
    print()

    tests = [
        ("Health Check", test_health_check),
        ("Prediction", test_prediction),
        ("Invalid City", test_invalid_city)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1

    print(f"\n{'='*50}")
    print("📊 Test Results Summary:" )   
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the API implementation.")

    print(f"⏰ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()