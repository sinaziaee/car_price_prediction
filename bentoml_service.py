import bentoml
import json
from pathlib import Path
import joblib
import pandas as pd
from src.processor import Processor


# Load the processor and model
model_dir = Path(__file__).parent / "models"
processor: Processor = joblib.load(model_dir / "processor.joblib")
processor.model = joblib.load(model_dir / "linear_regression_model.joblib")

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
            # Prepare input for prediction
            prepared_input = processor.prepare_data_for_inference(data, X_train_ref)
            
            # Make prediction
            predicted_price = processor.infer(prepared_input)
            
            return {
                "predicted_price": predicted_price,
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
            "service": "CarPricePrediction"
        }
