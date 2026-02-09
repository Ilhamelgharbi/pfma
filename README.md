# 🏠 SalesHouses - Moroccan Real Estate Price Prediction

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)

> **SalesHouses** is an AI-powered real estate price prediction platform for the Moroccan market, combining machine learning, modern web technologies, and comprehensive data analysis to provide accurate apartment price estimates.

## 🌟 Features

### 🤖 Advanced Machine Learning
- **Gradient Boosting Regressor** with 82.7% R² accuracy
- **Comprehensive feature engineering** (surface ratios, equipment scoring, city encoding)
- **Robust outlier detection** using IQR method
- **Cross-validation** with 5-fold evaluation

### 🏗️ Modern Architecture
- **FastAPI backend** for high-performance REST API
- **Streamlit frontend** for intuitive user interface
- **Docker containerization** for easy deployment
- **Supervisord** process management

### 📊 Rich Analytics & Visualizations
- **Interactive price predictions** with confidence intervals
- **Comprehensive model evaluation** metrics (MAE, RMSE, MAPE)
- **Data exploration** visualizations (correlations, distributions, outliers)
- **City-wise statistics** and market analysis

### 🎯 Smart Features
- **Real-time predictions** based on apartment characteristics
- **Multi-city support** across major Moroccan cities
- **Equipment-based pricing** adjustments
- **Confidence intervals** for prediction reliability

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB RAM minimum
- 2GB disk space

### One-Command Setup
```bash
# Clone the repository
git clone <repository-url>
cd pfma

# Build and run with Docker
docker build -t saleshouses .
docker run -p 7860:7860 -p 8000:8000 saleshouses
```

Access the application:
- **Frontend**: http://localhost:7860
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Local Development

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
# or with uv (recommended)
pip install uv
uv sync
```

### Running Components

#### Backend (FastAPI)
```bash
cd backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (Streamlit)
```bash
cd frontend
uv run streamlit run app.py --server.port=7860 --server.address=0.0.0.0
```

#### Training Script
```bash
cd notebooks
uv run python script.py
```

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.827 | Excellent (82.7% variance explained) |
| **MAE** | 185,807 MAD | Average prediction error |
| **RMSE** | 273,802 MAD | Root mean squared error |
| **MAPE** | 18.13% | Mean absolute percentage error |

### Feature Importance (Top 5)
1. **Surface Area** - Most influential factor
2. **City Location** - Geographic pricing variations
3. **Equipment Score** - Added value from amenities
4. **Number of Rooms** - Size indicator
5. **Bath-to-Room Ratio** - Quality metric

## 🏗️ Project Structure

```
pfma/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── files.py            # File handling utilities
│   └── test_api.py         # API tests
├── frontend/               # Streamlit frontend
│   └── app.py              # Main application
├── notebooks/              # ML pipeline & analysis
│   └── script.py           # Complete ML workflow
├── data/                   # Data management
│   ├── appartements-data-db.csv    # Raw dataset
│   └── processed/          # Cleaned data
├── models/                 # ML artifacts
│   ├── best_model.pkl      # Trained model
│   ├── scaler.pkl          # Feature scaler
│   └── metadata.json       # Model metadata
├── visualizations/         # Generated plots
├── reports/                # Model metrics & logs
├── rapport/                # Academic report figures
├── Dockerfile             # Container configuration
├── pyproject.toml         # Python dependencies
└── README.md              # This file
```

## 🔌 API Documentation

### Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
  "city": "Casablanca",
  "surface_area": 100,
  "nb_baths": 2,
  "total_rooms": 3,
  "equipment_list": ["Ascenseur", "Balcon"]
}
```

**Response:**
```json
{
  "predicted_price": 1179812,
  "price_per_m2": 11798,
  "confidence_interval": {
    "lower": 1002840,
    "upper": 1356784
  }
}
```

### Health Check
```http
GET /health
```

## 📈 Data Pipeline

1. **Data Collection** - Moroccan real estate listings
2. **Preprocessing** - Cleaning, feature engineering, encoding
3. **Outlier Detection** - IQR method for price anomalies
4. **Model Training** - Gradient Boosting with hyperparameter tuning
5. **Validation** - 5-fold cross-validation
6. **Deployment** - Docker containerization

### Key Features Engineered
- **Total Rooms**: Bedrooms + living rooms
- **Bath-to-Room Ratio**: Quality indicator
- **Surface per Room**: Space efficiency metric
- **Equipment Score**: Amenity value calculation
- **City Encoding**: One-hot encoding for locations

## 🎨 Visualizations Generated

The system automatically generates comprehensive visualizations:

### Model Performance
- **Training vs Test Metrics** comparison
- **Feature Importance** ranking
- **Residual Analysis** plots

### Data Analysis
- **Price Distribution** histograms
- **City-wise Statistics** bar charts
- **Correlation Matrix** heatmaps
- **Outlier Detection** before/after plots

### Academic Report Figures
- **Figure 9**: Price distribution analysis
- **Figure 10**: City price comparisons
- **Figure 12**: Outlier detection boxplots
- **Figure 20**: Model comparison metrics
- **Figure 22**: Performance evaluation dashboard

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_URL=http://localhost:8000

# Model Parameters
MODEL_PATH=models/best_model.pkl
SCALER_PATH=models/scaler.pkl
```

### Docker Configuration
- **Base Image**: `python:3.13-slim`
- **Package Manager**: `uv` for fast dependency resolution
- **Process Manager**: `supervisord` for multi-service orchestration
- **Ports**: 7860 (Streamlit), 8000 (FastAPI)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure Docker compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Ilham El Gharbi** 
- **Soukaina** 
- **Zeineb** X

## 🙏 Acknowledgments

- **Data Source**: Moroccan real estate market data
- **ML Framework**: Scikit-learn community
- **Web Frameworks**: FastAPI & Streamlit communities
- **Containerization**: Docker ecosystem

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the model performance metrics

---

**Built with ❤️ for the Moroccan real estate market**

*Last updated: February 2026*
