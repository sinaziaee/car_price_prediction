import argparse
import sys
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
    """Train and evaluate the model."""
    print(f"🚀 Starting training and evaluation")
    
    # Load processor and train model
    processor = Processor.load_from_preprocessed(args.data_dir)
    processor.train_and_evaluate_from_files(args.data_dir)
    
    # Save model
    processor.save_model(args.model_dir)


def infer(args):
    """Run inference on new data."""
    print(f"🔮 Running inference")
    
    # Load processor and run inference
    processor = Processor.load_from_model(args.model_dir)
    predicted_price = processor.infer_from_json(args.input_file, args.data_dir)
    
    print(f"💰 Predicted Price: ${predicted_price:,.2f}")
    
    # Save output if requested
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
        default='models',
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
