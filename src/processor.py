import joblib
import polars as pl
from .utils import prepare_input_for_prediction, scale_numerical_columns, scale_target, target_encode_nominal_columns, onehot_encode_nominal_columns
from .mlflow_utils import MLflowConfig, CarPriceMLflowModel, register_model_to_registry
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import tempfile
from pathlib import Path


class Processor():
    def __init__(self, dataset_path, columns_to_drop, categorical_columns, nominal_columns, remaining_nominal_columns, numerical_columns, mlflow_experiment_name="car-price-prediction"):
        self.dataset_path = dataset_path
        self.columns_to_drop = columns_to_drop
        self.categorical_columns = categorical_columns
        self.nominal_columns = nominal_columns
        self.remaining_nominal_columns = remaining_nominal_columns
        self.numerical_columns = numerical_columns
        self.df = self.read_dataset()
        
        # Initialize MLflow
        self.mlflow_config = MLflowConfig(mlflow_experiment_name)
        
        if self.df is None:
            raise ValueError(f"Failed to load dataset from {dataset_path}")
            
        self.min_max_scalers = None
        self.target_scaler = None
        self.unique_values_dict = None
        self.target_encoders = None
        self.onehot_encoders = None

        self.model = None

    def read_dataset(self) -> pl.DataFrame:
        try:
            df = pl.read_csv(self.dataset_path)
            if df is None or df.is_empty():
                raise ValueError(f"Dataset is empty or could not be read from {self.dataset_path}")
            return df
        except Exception as e:
            raise ValueError(f"Failed to read dataset from {self.dataset_path}: {str(e)}")

    def fix_null_values(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = self.df.drop(self.columns_to_drop)
        # drop nan categorical columns
        df = df.drop_nulls(subset=self.categorical_columns + ["price"])
        target = df["price"]
        data = df.drop("price")
        # add engineered features
        data = data.with_columns([
            (data["miles"] / (2023 - data["year"])).cast(pl.Int64).alias("miles_per_year"),
            (2023 - data["year"]).alias("age")
        ])
        # handle null values in numerical columns by filling with median
        data = data.with_columns([
            pl.col(col).fill_null(data[col].median()).alias(col)
            for col in self.numerical_columns
        ])
        
        return (data, target)
    
    def scale_numerical_columns(self, data: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
        data, min_max_scalers = scale_numerical_columns(data, self.numerical_columns)
        target, target_scaler = scale_target(target)

        self.min_max_scalers = min_max_scalers
        self.target_scaler = target_scaler

        target = pl.from_numpy(target.flatten()) 

        return data, target
    
    def find_unique_values(self, data: pl.DataFrame) -> None:
        unique_values_dict = {}
        for col in self.categorical_columns:
            unique_values = data[col].unique()
            unique_values_dict[col] = unique_values
        self.unique_values_dict = unique_values_dict


    def encode_categorical_features(self, data: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
        # convert nominal columns to target encoded columns
        target_encoded_data, target_encoders = target_encode_nominal_columns(data, target, self.nominal_columns)
        self.target_encoders = target_encoders
        # one-hot encode "transmission" and "drivetrain"
        onehot_encoded_data, onehot_encoders = onehot_encode_nominal_columns(target_encoded_data, self.remaining_nominal_columns)
        self.onehot_encoders = onehot_encoders

        return onehot_encoded_data
    
    def split_data(self, data: pl.DataFrame | pd.DataFrame, target: pl.DataFrame | pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        from sklearn.model_selection import train_test_split
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            target = target.to_pandas()
        else:
            pass
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
        return pl.from_pandas(X_train), pl.from_pandas(X_test), pl.from_pandas(y_train), pl.from_pandas(y_test)

    def train(self, X_train: pl.DataFrame, y_train: pl.DataFrame, log_to_mlflow: bool = True) -> LinearRegression:
        model = LinearRegression()
        self.model = model
        
        if log_to_mlflow:
            # Log model parameters
            model_params = {
                "algorithm": "LinearRegression",
                "fit_intercept": model.fit_intercept,
                "training_samples": len(X_train),
                "features": X_train.shape[1]
            }
            self.mlflow_config.log_model_params(model_params)
        
        model.fit(X_train, y_train)
        return model
    
    def score(self, X_test: pl.DataFrame, y_test: pl.DataFrame, log_to_mlflow: bool = True) -> Dict[str, float]:
        import numpy as np
        from sklearn.metrics import mean_absolute_error
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        }
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R^2 Score: {r2:.4f}")
        
        if log_to_mlflow:
            self.mlflow_config.log_metrics(metrics)
        
        return metrics

    def prepare_data_for_inference(self, inp: dict | pl.DataFrame, original_df:pl.DataFrame) -> pd.DataFrame:
        prepared_input = prepare_input_for_prediction(inp, numerical_columns=self.numerical_columns,
                                target_encoders=self.target_encoders, onehot_encoders=self.onehot_encoders, 
                                min_max_scalers=self.min_max_scalers, nominal_columns=self.nominal_columns, 
                                remaining_nominal_columns=self.remaining_nominal_columns,
                                numerical_medians={col: original_df[col].median() for col in self.numerical_columns})
        # Align prepared input columns and order with training features
        # Reindex will add any missing training columns as zeros and drop extras
        prepared_input = prepared_input.reindex(columns=original_df.columns, fill_value=0)
        # Convert to numeric (float) to match training dtypes used by the model
        prepared_input = prepared_input.astype(float)
        return prepared_input
    
    def preprocess(self):
        data, target = self.fix_null_values()
        data, target = self.scale_numerical_columns(data, target)
        self.find_unique_values(data)
        data = self.encode_categorical_features(data, target)
        X_train, X_test, y_train, y_test = self.split_data(data, target)
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train: pl.DataFrame, y_train: pl.DataFrame, X_test: pl.DataFrame, y_test: pl.DataFrame, 
                          run_name: str = None, log_to_mlflow: bool = True) -> Dict[str, float]:
        
        if log_to_mlflow:
            with self.mlflow_config.start_run(run_name=run_name):
                # Log preprocessing parameters
                preprocessing_params = {
                    "columns_dropped": len(self.columns_to_drop),
                    "categorical_columns": len(self.categorical_columns),
                    "numerical_columns": len(self.numerical_columns),
                    "nominal_columns": len(self.nominal_columns),
                    "dataset_path": str(self.dataset_path)
                }
                self.mlflow_config.log_preprocessing_params(preprocessing_params)
                
                model = self.train(X_train, y_train, log_to_mlflow=True)
                metrics = self.score(X_test, y_test, log_to_mlflow=True)
                
                # Log model
                self.mlflow_config.log_model(model, "linear_regression_model", 
                                           input_example=X_train.to_pandas().iloc[:5],
                                           signature=mlflow.models.infer_signature(
                                               X_train.to_pandas(), 
                                               y_train.to_pandas()
                                           ))
                
                # Log additional artifacts
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Save processor
                    joblib.dump(self, temp_path / "processor.joblib")
                    mlflow.log_artifact(str(temp_path / "processor.joblib"), "model_artifacts")
                    
                    # Save feature names
                    feature_names = list(X_train.columns)
                    with open(temp_path / "feature_names.txt", "w") as f:
                        f.write("\n".join(feature_names))
                    mlflow.log_artifact(str(temp_path / "feature_names.txt"), "model_artifacts")
                
                # Save model info for later registration
                self.last_mlflow_run_id = mlflow.active_run().info.run_id
                
                return metrics
        else:
            model = self.train(X_train, y_train, log_to_mlflow=False)
            metrics = self.score(X_test, y_test, log_to_mlflow=False)
            return metrics
            
        # save model for inference
        self.model = model
        # Note: Model saving is now handled in the CLI (main.py)
    
    def infer(self, df: pd.DataFrame) -> float:
        predicted_price_scaled = self.model.predict(df)
        predicted_price = self.target_scaler.inverse_transform(predicted_price_scaled).item()
        return int(predicted_price)
    
    def predict_single(self, input_dict: dict) -> float:
        """Predict a single car price from input dictionary (for MLflow model wrapper)."""
        # This method is used by the MLflow model wrapper
        # We need to load reference data for feature alignment
        # For now, we'll assume the input is properly formatted
        import pandas as pd
        
        # Convert dict to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Use existing inference pipeline (simplified for MLflow)
        predicted_price_scaled = self.model.predict(input_df)
        predicted_price = self.target_scaler.inverse_transform(predicted_price_scaled).item()
        return int(predicted_price)
    
    def save_preprocessed_data(self, output_dir):
        """Save preprocessed data and processor to specified directory."""
        from pathlib import Path
        import joblib
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess()
        
        # Save data
        X_train.write_csv(output_dir / "X_train.csv")
        X_test.write_csv(output_dir / "X_test.csv")
        y_train.write_csv(output_dir / "y_train.csv")
        y_test.write_csv(output_dir / "y_test.csv")
        
        # Save processor
        joblib.dump(self, output_dir / "processor.joblib")
        
        print(f"✅ Preprocessing complete! Data saved to {output_dir}/")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Test samples: {len(X_test)}")
        print(f"   - Features: {X_train.shape[1]}")
    
    @classmethod
    def load_from_preprocessed(cls, data_dir):
        """Load processor from preprocessed data directory."""
        import joblib
        from pathlib import Path
        
        data_dir = Path(data_dir)
        return joblib.load(data_dir / "processor.joblib")
    
    def train_and_evaluate_from_files(self, data_dir, run_name: str = None, log_to_mlflow: bool = True):
        """Load preprocessed data and train/evaluate model with MLflow tracking."""
        from pathlib import Path
        
        data_dir = Path(data_dir)
        
        # Load preprocessed data
        print("📂 Loading preprocessed data...")
        X_train = pl.read_csv(data_dir / "X_train.csv")
        X_test = pl.read_csv(data_dir / "X_test.csv")
        y_train = pl.read_csv(data_dir / "y_train.csv")
        y_test = pl.read_csv(data_dir / "y_test.csv")
        
        # Train and evaluate
        print("🎯 Training model...")
        metrics = self.train_and_evaluate(X_train, y_train, X_test, y_test, 
                                        run_name=run_name, log_to_mlflow=log_to_mlflow)
        
        return metrics
    
    def save_model(self, model_dir, register_to_mlflow: bool = True, model_name: str = "car-price-model", stage: str = "Staging"):
        """Save trained model to specified directory and optionally register to MLflow."""
        import joblib
        from pathlib import Path
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Traditional model saving
        joblib.dump(self.model, model_dir / "linear_regression_model.joblib")
        joblib.dump(self, model_dir / "processor.joblib")
        
        print(f"✅ Training complete! Model saved to {model_dir}/")
        
        # MLflow model registration
        if register_to_mlflow and hasattr(self, 'last_mlflow_run_id'):
            try:
                model_uri = f"runs:/{self.last_mlflow_run_id}/linear_regression_model"
                model_version = register_model_to_registry(model_uri, model_name, stage)
                if model_version:
                    print(f"🚀 Model registered to MLflow Registry as {model_name} v{model_version.version} in {stage}")
            except Exception as e:
                print(f"⚠️ Failed to register model to MLflow Registry: {e}")
    
    @classmethod
    def load_from_model(cls, model_dir):
        """Load processor and model from model directory."""
        import joblib
        from pathlib import Path
        
        model_dir = Path(model_dir)
        
        print("📂 Loading model and processor...")
        processor = joblib.load(model_dir / "processor.joblib")
        processor.model = joblib.load(model_dir / "linear_regression_model.joblib")
        
        return processor
    
    def infer_from_json(self, input_file, data_dir):
        """Run inference from JSON input file."""
        import json
        from pathlib import Path
        
        # Load input data
        print(f"📄 Loading input from {input_file}")
        with open(input_file, 'r') as f:
            input_dict = json.load(f)
        
        print(f"📊 Input data: {input_dict}")
        
        # Load original training data for feature alignment
        data_path = Path(data_dir) / "X_train.csv"
        original_df = pl.read_csv(data_path)
        
        # Prepare and run inference
        prepared_data = self.prepare_data_for_inference(input_dict, original_df)
        predicted_price = self.infer(prepared_data)
        
        return predicted_price 