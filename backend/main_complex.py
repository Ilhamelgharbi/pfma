"""
🏠 SalesHouses - FastAPI Backend
API for Moroccan Real Estate Price Prediction

Features:
- Health check endpoint
- Apartment price prediction endpoint
- Input validation with Pydantic models
- Proper error handling
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class ApartmentFeatures(BaseModel):
    """Input model for apartment features"""
    city: str = Field(..., description="City name (e.g., 'Casablanca', 'Rabat', 'Marrakech')")
    surface_area: float = Field(..., gt=0, le=500, description="Surface area in square meters")
    nb_baths: int = Field(..., ge=0, le=10, description="Number of bathrooms")
    total_rooms: int = Field(..., ge=1, le=15, description="Total number of rooms (bedrooms + living rooms)")
    equipment_list: Optional[List[str]] = Field(default=[], description="List of equipment features")

    @validator('city')
    def validate_city(cls, v):
        """Validate that city is supported"""
        if not hasattr(cls, '_available_cities'):
            # Load available cities from metadata
            try:
                with open('../models/metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    cls._available_cities = set(metadata.get('available_cities', []))
            except:
                cls._available_cities = set()

        if v not in cls._available_cities:
            raise ValueError(f"City '{v}' is not supported. Available cities: {sorted(cls._available_cities)}")
        return v

    @validator('equipment_list')
    def validate_equipment(cls, v):
        """Validate equipment list"""
        if not hasattr(cls, '_equipment_features'):
            try:
                with open('../models/metadata.json', 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    cls._equipment_features = set(metadata.get('equipment_features', []))
            except:
                cls._equipment_features = set()

        invalid_equipment = set(v) - cls._equipment_features
        if invalid_equipment:
            raise ValueError(f"Invalid equipment: {invalid_equipment}. Available: {sorted(cls._equipment_features)}")
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    predicted_price: float = Field(..., description="Predicted apartment price in DH")
    price_per_m2: float = Field(..., description="Predicted price per square meter in DH/m²")
    city: str = Field(..., description="City name")
    surface_area: float = Field(..., description="Surface area in m²")
    nb_baths: int = Field(..., description="Number of bathrooms")
    total_rooms: int = Field(..., description="Total number of rooms")
    equipment: List[str] = Field(..., description="List of equipment features")
    confidence_interval: Dict[str, float] = Field(..., description="Price confidence interval (lower, upper)")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    model_version: str = Field(..., description="Model version used")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    scaler_loaded: bool = Field(..., description="Whether scaler is loaded")
    metadata_loaded: bool = Field(..., description="Whether metadata is loaded")
    available_cities: List[str] = Field(..., description="List of supported cities")
    equipment_features: List[str] = Field(..., description="List of available equipment features")

# ============================================================================
# PREDICTION SERVICE
# ============================================================================

class PredictionService:
    """Service class for apartment price prediction"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.numeric_features = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load ML model, scaler, and metadata"""
        try:
            # Load model
            self.model = joblib.load('../models/best_model.pkl')
            logger.info("✅ ML model loaded successfully")

            # Load scaler
            self.scaler = joblib.load('../models/scaler.pkl')
            logger.info("✅ Scaler loaded successfully")

            # Load metadata
            with open('../models/metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info("✅ Metadata loaded successfully")

            # Extract numeric features
            self.numeric_features = self.metadata.get('numeric_features', [])

        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {str(e)}")
            raise RuntimeError(f"Failed to load ML artifacts: {str(e)}")

    def predict_price(self, features: ApartmentFeatures) -> PredictionResponse:
        """
        Predict apartment price based on input features

        Args:
            features: ApartmentFeatures object with apartment details

        Returns:
            PredictionResponse with predicted price and details
        """
        try:
            # Create feature vector
            feature_dict = self._create_feature_vector(features)

            # Convert to DataFrame
            X_pred = pd.DataFrame([feature_dict])

            # Ensure all features are present
            for feature in self.metadata['feature_names']:
                if feature not in X_pred.columns:
                    X_pred[feature] = 0

            # Reorder columns to match training order
            X_pred = X_pred[self.metadata['feature_names']]

            # Scale numeric features
            X_pred[self.numeric_features] = self.scaler.transform(X_pred[self.numeric_features])

            # Make prediction
            predicted_price = float(self.model.predict(X_pred)[0])

            # Calculate price per m²
            price_per_m2 = predicted_price / features.surface_area

            # Create confidence interval (85% - 115% of predicted price)
            confidence_interval = {
                'lower': predicted_price * 0.85,
                'upper': predicted_price * 1.15
            }

            # Create response
            response = PredictionResponse(
                predicted_price=round(predicted_price, 2),
                price_per_m2=round(price_per_m2, 2),
                city=features.city,
                surface_area=features.surface_area,
                nb_baths=features.nb_baths,
                total_rooms=features.total_rooms,
                equipment=features.equipment_list,
                confidence_interval=confidence_interval,
                prediction_timestamp=datetime.now().isoformat(),
                model_version=self.metadata.get('version', '1.0')
            )

            logger.info(f"✅ Prediction completed for {features.city}, {features.surface_area}m²: {predicted_price:,.0f} DH")
            return response

        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

    def _create_feature_vector(self, features: ApartmentFeatures) -> Dict[str, Any]:
        """
        Create feature vector from input features

        Args:
            features: ApartmentFeatures object

        Returns:
            Dictionary with all feature values
        """
        feature_dict = {}

        # Basic numeric features
        feature_dict['surface_area'] = features.surface_area
        feature_dict['nb_baths'] = features.nb_baths
        feature_dict['total_rooms'] = features.total_rooms

        # Engineered features
        feature_dict['bath_room_ratio'] = features.nb_baths / (features.total_rooms + 1)
        feature_dict['surface_per_room'] = features.surface_area / (features.total_rooms + 1)
        feature_dict['equipment_score'] = len(features.equipment_list)

        # City one-hot encoding
        city_col = f'city_{features.city}'
        for city_feature in self.metadata['city_features']:
            feature_dict[city_feature] = 1 if city_feature == city_col else 0

        # Equipment one-hot encoding
        for equip_feature in self.metadata['equipment_features']:
            feature_dict[equip_feature] = 1 if equip_feature in features.equipment_list else 0

        return feature_dict

    def get_health_status(self) -> HealthResponse:
        """Get service health status"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=self.model is not None,
            scaler_loaded=self.scaler is not None,
            metadata_loaded=self.metadata is not None,
            available_cities=self.metadata.get('available_cities', []) if self.metadata else [],
            equipment_features=self.metadata.get('equipment_features', []) if self.metadata else []
        )

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="SalesHouses - Moroccan Real Estate Price Prediction API",
    description="API for predicting apartment prices in Morocco using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns service status and loaded components information
    """
    return prediction_service.get_health_status()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_apartment_price(features: ApartmentFeatures):
    """
    Predict apartment price based on input features

    This endpoint takes apartment features and returns a predicted price
    along with confidence intervals and additional metadata.
    """
    return prediction_service.predict_price(features)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "🏠 SalesHouses - Moroccan Real Estate Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": True,
        "message": "Internal server error",
        "status_code": 500
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("🚀 Starting SalesHouses FastAPI application")
    logger.info("📊 Loading ML artifacts...")

    try:
        # Artifacts are loaded in PredictionService __init__
        health = prediction_service.get_health_status()
        if health.model_loaded and health.scaler_loaded and health.metadata_loaded:
            logger.info("✅ All ML artifacts loaded successfully")
            logger.info(f"🏙️  Available cities: {len(health.available_cities)}")
            logger.info(f"🔧 Equipment features: {len(health.equipment_features)}")
        else:
            logger.error("❌ Failed to load ML artifacts")
            raise RuntimeError("ML artifacts not loaded")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)