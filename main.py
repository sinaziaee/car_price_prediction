import argparse
import sys
import joblib
import polars as pl
from pathlib import Path

from src.processor import Processor


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


    # Initialize processor
    processor = Processor(
        dataset_path=args.dataset,
        columns_to_drop=columns_to_drop,
        categorical_columns=categorical_columns,
        nominal_columns=nominal_columns,
        remaining_nominal_columns=remaining_nominal_columns,
        numerical_columns=numerical_columns
    )
    
    # Preprocess
    X_train, X_test, y_train, y_test = processor.preprocess()
    
    # Save preprocessed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.write_csv(output_dir / "X_train.csv")
    X_test.write_csv(output_dir / "X_test.csv")
    y_train.write_csv(output_dir / "y_train.csv")
    y_test.write_csv(output_dir / "y_test.csv")
    
    # Save processor for later use
    joblib.dump(processor, output_dir / "processor.joblib")
    
    print(f"✅ Preprocessing complete! Data saved to {output_dir}/")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Features: {X_train.shape[1]}")


def train_eval(args):
    """Train and evaluate the model."""
    print(f"🚀 Starting training and evaluation")
    
    data_dir = Path(args.data_dir)
    
    # Load preprocessed data
    print("📂 Loading preprocessed data...")
    X_train = pl.read_csv(data_dir / "X_train.csv")
    X_test = pl.read_csv(data_dir / "X_test.csv")
    y_train = pl.read_csv(data_dir / "y_train.csv")
    y_test = pl.read_csv(data_dir / "y_test.csv")
    
    # Load processor
    processor = joblib.load(data_dir / "processor.joblib")
    
    # Train and evaluate
    print("🎯 Training model...")
    processor.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Save model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(processor.model, model_dir / "linear_regression_model.joblib")
    joblib.dump(processor, model_dir / "processor.joblib")
    
    print(f"✅ Training complete! Model saved to {model_dir}/")


def infer(args):
    """Run inference on new data."""
    print(f"🔮 Running inference")
    
    model_dir = Path(args.model_dir)
    
    print("------------------", model_dir)

    # Load processor and model
    print("📂 Loading model and processor...")
    processor = joblib.load(model_dir / "processor.joblib")
    processor.model = joblib.load(model_dir / "linear_regression_model.joblib")
    
    # Load input data from JSON file
    import json
    print(f"📄 Loading input from {args.input_file}")
    with open(args.input_file, 'r') as f:
        input_dict = json.load(f)
    
    print(f"📊 Input data: {input_dict}")
    
    # Prepare data for inference
    original_df = pl.read_csv(model_dir.parent.parent / args.data_dir / "X_train.csv")
    prepared_data = processor.prepare_data_for_inference(input_dict, original_df)
    
    # Run inference
    predicted_price = processor.infer(prepared_data)
    
    print(f"💰 Predicted Price: ${predicted_price:,.2f}")
    
    if args.output_file:
        output_path = Path(args.output_file)
        with output_path.open('w') as f:
            f.write(f"Predicted Price: ${predicted_price:,.2f}\n")
        print(f"✅ Results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Car Price Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  uv run python main.py preprocess --dataset datasets/ca-dealers-used.csv
  
  # Train and evaluate
  uv run python main.py train-eval
  
  # Infer from JSON file
  uv run python main.py infer --input-file input.json
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
        default='src/models',
        help='Directory to save the trained model'
    )
    
    # Infer command
    infer_parser = subparsers.add_parser(
        'infer',
        help='Run inference on new data'
    )
    infer_parser.add_argument(
        '--model-dir',
        type=str,
        default='src/models',
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
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess(args)
    elif args.command == 'train-eval':
        train_eval(args)
    elif args.command == 'infer':
        infer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
