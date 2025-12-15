import argparse
import sys
from pathlib import Path
from src.processor import Processor
from src.mlflow_utils import get_best_model, load_model_from_registry
import mlflow


def preprocess(args):
    """Preprocess the dataset and save preprocessed data."""
    print(f"🔄 Starting preprocessing with dataset: {args.dataset}")
    
    # Define columns
    columns_to_drop = ["id", "vin", "stock_no", "seller_name", "street", "zip"]
    categorical_columns = ["make", "model", "trim", "body_type", "vehicle_type",
                        "drivetrain", "transmission", "fuel_type", "engine_block",
                        "city", "state"]
    numerical_columns = ["year", "miles", "miles_per_year", "age", "engine_size"]
    nominal_columns = ["make", "model", "trim", "body_type", "vehicle_type",
                    "fuel_type", "city", "state"]
    remaining_nominal_columns = ["transmission", "drivetrain", "engine_block"]

    # Initialize processor and run preprocessing
    processor = Processor(
        dataset_path=args.dataset,
        columns_to_drop=columns_to_drop,
        categorical_columns=categorical_columns,
        nominal_columns=nominal_columns,
        remaining_nominal_columns=remaining_nominal_columns,
        numerical_columns=numerical_columns
    )
    
    # Save preprocessed data
    processor.save_preprocessed_data(args.output_dir)


def train_eval(args):
    """Train and evaluate the model with MLflow tracking."""
    print(f"🚀 Starting training and evaluation")
    
    # Load processor and train model
    processor = Processor.load_from_preprocessed(args.data_dir)
    
    # Set up run name
    run_name = getattr(args, 'run_name', None)
    if not run_name:
        from datetime import datetime
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Train with MLflow tracking
    log_to_mlflow = not getattr(args, 'no_mlflow', False)
    metrics = processor.train_and_evaluate_from_files(
        args.data_dir, 
        run_name=run_name, 
        log_to_mlflow=log_to_mlflow
    )
    
    # Save model
    register_to_mlflow = log_to_mlflow and not getattr(args, 'no_registry', False)
    model_name = getattr(args, 'model_name', 'car-price-model')
    stage = getattr(args, 'stage', 'Staging')
    
    processor.save_model(
        args.model_dir, 
        register_to_mlflow=register_to_mlflow,
        model_name=model_name,
        stage=stage
    )
    
    if log_to_mlflow:
        print(f"📈 View experiment results: mlflow ui --host 0.0.0.0 --port 5000")
        print(f"🔗 Run ID: {getattr(processor, 'last_mlflow_run_id', 'N/A')}")


def mlflow_ui(args):
    """Start MLflow UI server."""
    import subprocess
    
    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 5000)
    
    print(f"🚀 Starting MLflow UI on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(["mlflow", "ui", "--host", host, "--port", str(port)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Error starting MLflow UI: {e}")
        sys.exit(1)


def mlflow_best_model(args):
    """Get the best model from MLflow experiments."""
    experiment_name = getattr(args, 'experiment_name', 'car-price-prediction')
    metric_name = getattr(args, 'metric', 'rmse')
    
    print(f"🔍 Finding best model in experiment '{experiment_name}' by {metric_name}...")
    
    best_run_id = get_best_model(experiment_name, metric_name, ascending=True)
    
    if best_run_id:
        print(f"🏆 Best model found: Run ID {best_run_id}")
        print(f"🔗 View in MLflow UI: http://localhost:5000/#/experiments/runs/{best_run_id}")
    else:
        print(f"⚠️ No models found in experiment '{experiment_name}'")


def mlflow_serve_model(args):
    """Serve a model from MLflow Model Registry."""
    import subprocess
    
    model_name = args.model_name
    stage = getattr(args, 'stage', 'Production')
    host = getattr(args, 'host', '127.0.0.1')
    port = getattr(args, 'port', 5001)
    
    model_uri = f"models:/{model_name}/{stage}"
    
    print(f"🚀 Serving model {model_name} (stage: {stage}) on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            "mlflow", "models", "serve",
            "-m", model_uri,
            "--host", host,
            "--port", str(port)
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Model server stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Error serving model: {e}")
        sys.exit(1)


def infer(args):
    """Run inference on new data."""
    print(f"🔮 Running inference")
    
    # Check if we should load from MLflow registry
    if hasattr(args, 'from_registry') and args.from_registry:
        model_name = getattr(args, 'model_name', 'car-price-model')
        stage = getattr(args, 'stage', 'Production')
        
        print(f"📦 Loading model from MLflow Registry: {model_name} ({stage})")
        mlflow_model = load_model_from_registry(model_name, stage)
        
        if mlflow_model:
            # Load input data
            import json
            with open(args.input_file, 'r') as f:
                input_data = json.load(f)
            
            # Make prediction using MLflow model
            import pandas as pd
            input_df = pd.DataFrame([input_data])
            result = mlflow_model.predict(input_df)
            
            predicted_price = result.iloc[0]['predicted_price'] if isinstance(result, pd.DataFrame) else result
            
            print(f"💰 Predicted price: ${predicted_price:,.2f}")
            
            # Save result if output file specified
            if getattr(args, 'output_file', None):
                output = {"predicted_price": float(predicted_price), "input": input_data}
                with open(args.output_file, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"✅ Results saved to {args.output_file}")
            
            return predicted_price
        else:
            print("⚠️ Failed to load model from MLflow Registry, falling back to local model...")


            print("⚠️ Failed to load model from MLflow Registry, falling back to local model...")
    
    # Traditional inference using local model
    processor = Processor.load_from_model(args.model_dir)
    predicted_price = processor.infer_from_json(args.input_file, args.data_dir)
    
    print(f"💰 Predicted price: ${predicted_price:,.2f}")
    
    # Save result if output file specified
    if getattr(args, 'output_file', None):
        import json
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
        
        output = {"predicted_price": float(predicted_price), "input": input_data}
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✅ Results saved to {args.output_file}")
    
    return predicted_price


def main():
    parser = argparse.ArgumentParser(
        description="Car Price Prediction CLI with MLflow Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  uv run python main.py preprocess --dataset datasets/ca-dealers-used.csv
  
  # Train and evaluate with MLflow
  uv run python main.py train-eval --run-name "experiment-1"
  
  # Start MLflow UI
  uv run python main.py mlflow-ui
  
  # Infer from local model
  uv run python main.py infer --input-file input.json
  
  # Infer from MLflow registry
  uv run python main.py infer --input-file input.json --from-registry
  
  # Serve model from MLflow
  uv run python main.py mlflow-serve --model-name car-price-model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess the dataset'
    )
    preprocess_parser.add_argument(
        '--dataset',
        type=str,
        default='datasets/ca-dealers-used.csv',
        help='Path to the dataset CSV file'
    )
    preprocess_parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save preprocessed data'
    )
    
    # Train-eval command
    train_eval_parser = subparsers.add_parser(
        'train-eval',
        help='Train and evaluate the model'
    )
    train_eval_parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing preprocessed data'
    )
    train_eval_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save the trained model'
    )
    train_eval_parser.add_argument(
        '--run-name',
        type=str,
        help='Name for the MLflow run (optional)'
    )
    train_eval_parser.add_argument(
        '--model-name',
        type=str,
        default='car-price-model',
        help='Name for model in MLflow registry (default: car-price-model)'
    )
    train_eval_parser.add_argument(
        '--stage',
        type=str,
        choices=['None', 'Staging', 'Production', 'Archived'],
        default='Staging',
        help='MLflow model registry stage (default: Staging)'
    )
    train_eval_parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )
    train_eval_parser.add_argument(
        '--no-registry',
        action='store_true',
        help='Skip MLflow model registry registration'
    )
    
    # Infer command
    infer_parser = subparsers.add_parser(
        'infer',
        help='Run inference on new data'
    )
    infer_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing the trained model'
    )
    infer_parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing training data (for column alignment)'
    )
    infer_parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='JSON file with input data (e.g., {"year": 2020, "miles": 30000, "brand": "Toyota", ...})'
    )
    infer_parser.add_argument(
        '--output-file',
        type=str,
        help='File to save prediction results'
    )
    infer_parser.add_argument(
        '--from-registry',
        action='store_true',
        help='Load model from MLflow Model Registry instead of local files'
    )
    infer_parser.add_argument(
        '--model-name',
        type=str,
        default='car-price-model',
        help='Name of model in MLflow registry (when using --from-registry)'
    )
    infer_parser.add_argument(
        '--stage',
        type=str,
        default='Production',
        help='Stage of model in MLflow registry (when using --from-registry)'
    )
    
    # MLflow UI command
    mlflow_ui_parser = subparsers.add_parser(
        'mlflow-ui',
        help='Start MLflow UI server'
    )
    mlflow_ui_parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host for MLflow UI (default: 0.0.0.0)'
    )
    mlflow_ui_parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for MLflow UI (default: 5000)'
    )
    
    # MLflow best model command
    mlflow_best_parser = subparsers.add_parser(
        'mlflow-best',
        help='Find the best model from MLflow experiments'
    )
    mlflow_best_parser.add_argument(
        '--experiment-name',
        type=str,
        default='car-price-prediction',
        help='Name of MLflow experiment (default: car-price-prediction)'
    )
    mlflow_best_parser.add_argument(
        '--metric',
        type=str,
        default='rmse',
        help='Metric to optimize (default: rmse)'
    )
    
    # MLflow serve model command
    mlflow_serve_parser = subparsers.add_parser(
        'mlflow-serve',
        help='Serve model from MLflow Model Registry'
    )
    mlflow_serve_parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of model in MLflow registry'
    )
    mlflow_serve_parser.add_argument(
        '--stage',
        type=str,
        default='Production',
        help='Stage of model to serve (default: Production)'
    )
    mlflow_serve_parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host for model server (default: 127.0.0.1)'
    )
    mlflow_serve_parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='Port for model server (default: 5001)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess(args)
    elif args.command == 'train-eval':
        train_eval(args)
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'mlflow-ui':
        mlflow_ui(args)
    elif args.command == 'mlflow-best':
        mlflow_best_model(args)
    elif args.command == 'mlflow-serve':
        mlflow_serve_model(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
