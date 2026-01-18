# 🏠 SalesHouses - FastAPI Backend

Moroccan Real Estate Price Prediction API built with FastAPI and machine learning.

## 🚀 Features

- **Health Check**: Monitor API status and loaded components
- **Price Prediction**: Predict apartment prices based on features
- **Input Validation**: Pydantic models with comprehensive validation
- **Error Handling**: Proper HTTP status codes and error messages
- **CORS Support**: Cross-origin resource sharing enabled
- **Interactive Docs**: Automatic API documentation at `/docs`

## 📋 Requirements

- Python 3.8+
- Trained ML model (`../models/best_model.pkl`)
- Scaler (`../models/scaler.pkl`)
- Metadata (`../models/metadata.json`)

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model artifacts exist:**
   ```bash
   ls ../models/
   # Should contain: best_model.pkl, scaler.pkl, metadata.json
   ```

## 🚀 Running the API

### Development Server
```bash
python main.py
```

### Production Server (with uvicorn)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📖 API Endpoints

### GET `/health`
Health check endpoint that returns service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-18T12:00:00",
  "model_loaded": true,
  "scaler_loaded": true,
  "metadata_loaded": true,
  "available_cities": ["Casablanca", "Rabat", "Marrakech", ...],
  "equipment_features": ["Ascenseur", "Balcon", "Parking", ...]
}
```

### POST `/predict`
Predict apartment price based on input features.

**Request Body:**
```json
{
  "city": "Casablanca",
  "surface_area": 100,
  "nb_baths": 2,
  "total_rooms": 3,
  "equipment_list": ["Ascenseur", "Balcon", "Parking"]
}
```

**Response:**
```json
{
  "predicted_price": 1179812.34,
  "price_per_m2": 11798.12,
  "city": "Casablanca",
  "surface_area": 100,
  "nb_baths": 2,
  "total_rooms": 3,
  "equipment": ["Ascenseur", "Balcon", "Parking"],
  "confidence_interval": {
    "lower": 1000840.0,
    "upper": 1356784.0
  },
  "prediction_timestamp": "2026-01-18T12:00:00",
  "model_version": "1.0"
}
```

## 🧪 Testing

### Run the test script:
```bash
python test_api.py
```

### Manual testing with curl:
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "city": "Casablanca",
       "surface_area": 100,
       "nb_baths": 2,
       "total_rooms": 3,
       "equipment_list": ["Ascenseur", "Balcon"]
     }'
```

### Interactive API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## 🏗️ Architecture

### Components:

1. **Data Models** (`ApartmentFeatures`, `PredictionResponse`, `HealthResponse`)
   - Pydantic models for input/output validation
   - Custom validators for city and equipment validation

2. **Prediction Service** (`PredictionService`)
   - Handles ML model loading and prediction logic
   - Feature engineering and preprocessing
   - Error handling and logging

3. **FastAPI Application**
   - REST API endpoints
   - Middleware for CORS
   - Error handlers
   - Startup validation

### Data Flow:
1. **Input Validation** → Pydantic models validate request
2. **Feature Engineering** → Service creates feature vector
3. **Preprocessing** → Scale numeric features
4. **Prediction** → ML model generates prediction
5. **Response** → Formatted JSON response

## 🔧 Configuration

### Model Artifacts Paths
The API expects model artifacts in the parent directory:
```
../models/
├── best_model.pkl      # Trained ML model
├── scaler.pkl          # Feature scaler
└── metadata.json       # Model metadata
```

### Environment Variables
```bash
# Optional: Set host and port
export HOST=0.0.0.0
export PORT=8000
```

## 🚨 Error Handling

### HTTP Status Codes:
- `200`: Success
- `422`: Validation error (invalid input)
- `500`: Internal server error

### Error Response Format:
```json
{
  "error": true,
  "message": "Detailed error message",
  "status_code": 500
}
```

## 📊 Supported Features

### Cities
The API supports cities that were present in the training data. Use the `/health` endpoint to get the current list.

### Equipment Features
- Ascenseur (Elevator)
- Balcon (Balcony)
- Chauffage (Heating)
- Climatisation (Air Conditioning)
- Concierge
- Cuisine Équipée (Equipped Kitchen)
- Duplex
- Meublé (Furnished)
- Parking
- Sécurité (Security)
- Terrasse (Terrace)

## 🔍 Monitoring

### Health Checks
- Model loading status
- Scaler availability
- Metadata validation
- Available cities and equipment

### Logging
The API logs all predictions and errors for monitoring and debugging.

## 🚀 Deployment

### Production Checklist:
- [ ] Set appropriate CORS origins
- [ ] Configure logging level
- [ ] Set up monitoring/alerting
- [ ] Use production WSGI server (gunicorn)
- [ ] Configure reverse proxy (nginx)
- [ ] Set up SSL/TLS certificates

### Docker Deployment:
```bash
# Build image
docker build -t saleshouses-api .

# Run container
docker run -p 8000:8000 saleshouses-api
```

## 🤝 Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Update tests for new features
4. Update documentation

## 📄 License

This project is part of the SalesHouses Moroccan Real Estate platform.