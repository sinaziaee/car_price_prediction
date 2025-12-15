"""
MLflow configuration and utilities for car price prediction project.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd


class MLflowConfig:
    """MLflow configuration and setup utilities."""
    
    def __init__(self, experiment_name: str = "car-price-prediction", tracking_uri: str = None):
        """
        Initialize MLflow configuration.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local file store)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (default to local mlruns directory)
        if tracking_uri is None:
            project_root = Path(__file__).parent.parent
            tracking_uri = f"file://{project_root}/mlruns"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)
        except Exception as e:
            print(f"Warning: Could not create/get experiment: {e}")
            self.experiment = None
    
    def start_run(self, run_name: str = None, nested: bool = False):
        """Start an MLflow run with the configured experiment."""
        if self.experiment:
            mlflow.set_experiment(self.experiment_name)
        
        return mlflow.start_run(run_name=run_name, nested=nested)
    
    @staticmethod
    def log_preprocessing_params(params: Dict[str, Any]):
        """Log preprocessing parameters."""
        for key, value in params.items():
            mlflow.log_param(f"preprocessing_{key}", value)
    
    @staticmethod
    def log_model_params(params: Dict[str, Any]):
        """Log model parameters."""
        for key, value in params.items():
            mlflow.log_param(f"model_{key}", value)
    
    @staticmethod
    def log_metrics(metrics: Dict[str, float]):
        """Log model performance metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    @staticmethod
    def log_artifacts(artifact_path: str, artifact_name: str = None):
        """Log artifacts (files/directories)."""
        mlflow.log_artifacts(artifact_path, artifact_name)
    
    @staticmethod
    def log_model(model, model_name: str, **kwargs):
        """Log a scikit-learn model."""
        mlflow.sklearn.log_model(model, model_name, **kwargs)


class CarPriceMLflowModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model wrapper for car price prediction."""
    
    def __init__(self):
        self.processor = None
    
    def load_context(self, context):
        """Load the processor and model from MLflow context."""
        import joblib
        processor_path = context.artifacts["processor"]
        self.processor = joblib.load(processor_path)
    
    def predict(self, context, model_input):
        """Make predictions using the processor."""
        if isinstance(model_input, pd.DataFrame):
            # Handle batch predictions
            predictions = []
            for _, row in model_input.iterrows():
                input_dict = row.to_dict()
                pred = self.processor.predict_single(input_dict)
                predictions.append(pred)
            return pd.DataFrame({"predicted_price": predictions})
        else:
            # Handle single prediction
            return self.processor.predict_single(model_input)


def register_model_to_registry(model_uri: str, model_name: str, stage: str = "Staging"):
    """
    Register a model to MLflow Model Registry.
    
    Args:
        model_uri: URI of the model in MLflow
        model_name: Name for the registered model
        stage: Stage to assign (None, Staging, Production, Archived)
    """
    try:
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition to specified stage
        if stage and stage != "None":
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            print(f"Model {model_name} version {model_version.version} transitioned to {stage}")
        
        return model_version
    
    except Exception as e:
        print(f"Failed to register model: {e}")
        return None


def load_model_from_registry(model_name: str, stage: str = "Production"):
    """
    Load a model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Stage to load from (Staging, Production, etc.)
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Failed to load model from registry: {e}")
        return None


def get_best_model(experiment_name: str, metric_name: str = "rmse", ascending: bool = True):
    """
    Get the best model from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to optimize
        ascending: True for metrics to minimize (like RMSE), False for metrics to maximize (like R2)
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) == 0:
            return None
        
        best_run = runs.iloc[0]
        return best_run.run_id
    
    except Exception as e:
        print(f"Failed to get best model: {e}")
        return None