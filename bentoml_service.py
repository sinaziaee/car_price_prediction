import bentoml
import json
import os
from pathlib import Path
import joblib
import pandas as pd
from src.processor import Processor
from src.mlflow_utils import load_model_from_registry
import mlflow


# Configuration for model source
USE_MLFLOW_REGISTRY = os.getenv("USE_MLFLOW_REGISTRY", "false").lower() == "true"
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "car-price-model")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# Try to load model from MLflow first, fallback to local files
processor = None
mlflow_model = None

if USE_MLFLOW_REGISTRY:
    print(f"🔄 Attempting to load model from MLflow Registry: {MLFLOW_MODEL_NAME} ({MLFLOW_MODEL_STAGE})")
    try:
        mlflow_model = load_model_from_registry(MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
        if mlflow_model:
            print(f"✅ Successfully loaded model from MLflow Registry")
        else:
            print(f"⚠️ Failed to load from MLflow Registry, falling back to local model")
    except Exception as e:
        print(f"⚠️ Error loading from MLflow Registry: {e}, falling back to local model")

if not mlflow_model:
    # Load local model
    print("🔄 Loading local model files")
    model_dir = Path(__file__).parent / "models"
    processor = joblib.load(model_dir / "processor.joblib")
    processor.model = joblib.load(model_dir / "linear_regression_model.joblib")
    print("✅ Successfully loaded local model")

# Get reference DataFrame for column alignment
X_train_path = Path(__file__).parent / "data" / "processed" / "X_train.csv"
if X_train_path.exists():
    import polars as pl
    X_train_pl = pl.read_csv(X_train_path)
    X_train_ref = X_train_pl.to_pandas()
else:
    X_train_ref = None


@bentoml.service
class CarPricePrediction:
    """BentoML service for car price prediction."""

    @bentoml.api
    def predict(self, data: dict) -> dict:
        """
        Predict car price from input features.
        
        Args:
            data: Dictionary containing car features like:
                {
                    "year": 2019,
                    "miles": 7120,
                    "make": "Acura",
                    "model": "NSX",
                    "trim": "...",
                    "body_type": "...",
                    "vehicle_type": "...",
                    "drivetrain": "...",
                    "transmission": "Automatic",
                    "fuel_type": "...",
                    "engine_block": "...",
                    "engine_size": ...,
                    "city": "...",
                    "state": "BC"
                }
        
        Returns:
            Dictionary with predicted price: {"predicted_price": 123456}
        """
        try:
            if mlflow_model:
                # Use MLflow model
                input_df = pd.DataFrame([data])
                result = mlflow_model.predict(input_df)
                predicted_price = result.iloc[0]['predicted_price'] if isinstance(result, pd.DataFrame) else result
            else:
                # Use local model
                prepared_input = processor.prepare_data_for_inference(data, X_train_ref)
                predicted_price = processor.infer(prepared_input)
            
            return {
                "predicted_price": int(predicted_price),
                "model_source": "mlflow_registry" if mlflow_model else "local_files",
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }

    @bentoml.api
    def health_check(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "CarPricePrediction",
            "model_source": "mlflow_registry" if mlflow_model else "local_files",
            "mlflow_model_name": MLFLOW_MODEL_NAME if mlflow_model else None,
            "mlflow_model_stage": MLFLOW_MODEL_STAGE if mlflow_model else None
        }
