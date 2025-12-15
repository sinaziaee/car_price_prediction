# Car Price Prediction Service

A machine learning service for predicting car prices with automated CI/CD pipeline.

## Features

- ✅ Data preprocessing pipeline
- ✅ Linear regression model training
- ✅ CLI interface for all operations
- ✅ Dockerized deployment
- ✅ BentoML service with REST API
- ✅ GitHub Actions CI/CD pipeline
- ✅ MLFlow experiment tracking and model registry
- ✅ MLFlow UI for experiment visualization
- ✅ Automated model versioning and deployment
- 🚧 AWS ECS/Lambda deployment (planned)
- 🚧 Web scraping for data collection (planned)
- 🚧 RAG with Gemini and Langchain (planned)

## CI/CD Pipeline

This project includes automated GitHub Actions workflows:

### Continuous Integration (CI)
- **Trigger**: Every push and pull request
- **Actions**: 
  - Python dependency installation
  - Code linting with flake8
  - Data preprocessing tests
  - Model training validation  
  - Inference testing
  - Docker image building

### Continuous Deployment (CD)
- **Trigger**: Push to `main` branch
- **Actions**:
  - Build and tag Docker image
  - Push to GitHub Container Registry
  - Deploy to staging environment
  - Optional AWS ECS deployment

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Preprocess data:**
   ```bash
   uv run python main.py preprocess --dataset datasets/ca-dealers-used.csv --output-dir data/processed
   ```

3. **Train model:**
   ```bash
   uv run python main.py train-eval --data-dir data/processed --model-dir models
   ```

4. **Start MLflow UI:**
   ```bash
   uv run mlflow ui --host 0.0.0.0 --port 5000
   ```

5. **Run inference:**
   ```bash
   uv run python main.py infer --input-file input.json --model-dir models --data-dir data/processed
   ```

6. **Start BentoML service:**
   ```bash
   uv run bentoml serve bentoml_service:CarPricePrediction --production
   ```

### Docker Deployment

1. **Start Docker:**
   ```bash
   sudo systemctl start docker
   ```

2. **Build image:**
   ```bash
   docker build -t car-price:latest .
   ```

3. **Run as service:**
   ```bash
   docker run -p 3000:3000 --rm car-price:latest
   ```

4. **Run interactively (for development/testing):**
   ```bash
   docker run -p 3000:3000 -it --rm --entrypoint /bin/bash car-price:latest
   ```

### Testing the API

Test the prediction endpoint:
```bash
curl -X POST "http://localhost:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2019,
    "miles": 7120,
    "make": "Acura",
    "model": "NSX",
    "trim": "Base",
    "body_type": "Coupe",
    "vehicle_type": "Car",
    "drivetrain": "AWD",
    "transmission": "Automatic",
    "fuel_type": "Gasoline",
    "engine_block": "V6",
    "engine_size": 3.5,
    "city": "Vancouver",
    "state": "BC"
  }'
```

## Development

### Running Tests
```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v
```

### Code Quality
```bash
# Run linting
uv run flake8 src/ main.py bentoml_service.py --max-line-length=100
```


