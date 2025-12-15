#!/bin/bash

# MLflow Setup Script for Car Price Prediction Project
# This script sets up MLflow tracking server and initializes the experiment

set -e  # Exit on any error

echo "🚀 Setting up MLflow for Car Price Prediction Project"

# Default configuration
MLFLOW_HOST=${MLFLOW_HOST:-"0.0.0.0"}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"car-price-prediction"}
BACKEND_STORE_URI=${BACKEND_STORE_URI:-"sqlite:///mlflow.db"}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"./mlruns"}

# Function to check if MLflow is installed
check_mlflow() {
    if ! command -v mlflow &> /dev/null; then
        echo "❌ MLflow is not installed. Please install it first:"
        echo "   uv add mlflow"
        exit 1
    fi
    echo "✅ MLflow is installed"
}

# Function to set up MLflow tracking
setup_tracking() {
    echo "🔧 Setting up MLflow tracking..."
    
    # Create mlruns directory if it doesn't exist
    mkdir -p mlruns
    
    # Set environment variables
    export MLFLOW_TRACKING_URI="$BACKEND_STORE_URI"
    export MLFLOW_DEFAULT_ARTIFACT_ROOT="$ARTIFACT_ROOT"
    
    echo "📊 MLflow Tracking URI: $MLFLOW_TRACKING_URI"
    echo "📁 Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
}

# Function to create experiment
create_experiment() {
    echo "🧪 Creating MLflow experiment: $EXPERIMENT_NAME"
    
    python3 -c "
import mlflow
import os

mlflow.set_tracking_uri('$BACKEND_STORE_URI')

try:
    experiment = mlflow.get_experiment_by_name('$EXPERIMENT_NAME')
    if experiment is None:
        experiment_id = mlflow.create_experiment('$EXPERIMENT_NAME')
        print(f'✅ Created experiment: $EXPERIMENT_NAME (ID: {experiment_id})')
    else:
        print(f'✅ Experiment already exists: $EXPERIMENT_NAME (ID: {experiment.experiment_id})')
except Exception as e:
    print(f'❌ Error creating experiment: {e}')
    exit(1)
"
}

# Function to start MLflow UI (background process)
start_ui() {
    echo "🌐 Starting MLflow UI..."
    echo "📍 UI will be available at: http://$MLFLOW_HOST:$MLFLOW_PORT"
    echo "🛑 Press Ctrl+C to stop"
    
    # Set tracking URI for UI
    export MLFLOW_TRACKING_URI="$BACKEND_STORE_URI"
    
    # Start MLflow UI
    mlflow ui --host "$MLFLOW_HOST" --port "$MLFLOW_PORT" --backend-store-uri "$BACKEND_STORE_URI"
}

# Function to show usage
usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup    Set up MLflow experiment (default)"
    echo "  ui       Start MLflow UI server"
    echo "  help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MLFLOW_HOST           Host for MLflow UI (default: 0.0.0.0)"
    echo "  MLFLOW_PORT           Port for MLflow UI (default: 5000)"
    echo "  EXPERIMENT_NAME       MLflow experiment name (default: car-price-prediction)"
    echo "  BACKEND_STORE_URI     MLflow backend store URI (default: sqlite:///mlflow.db)"
    echo "  ARTIFACT_ROOT         MLflow artifact root (default: ./mlruns)"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Set up MLflow experiment"
    echo "  $0 ui                       # Start MLflow UI"
    echo "  MLFLOW_PORT=5001 $0 ui      # Start UI on port 5001"
}

# Main script logic
main() {
    case "${1:-setup}" in
        setup)
            check_mlflow
            setup_tracking
            create_experiment
            echo ""
            echo "🎉 MLflow setup complete!"
            echo "📊 To view experiments: $0 ui"
            echo "🚀 To train with tracking: python main.py train-eval --data-dir data/processed --model-dir models"
            ;;
        ui)
            check_mlflow
            setup_tracking
            start_ui
            ;;
        help)
            usage
            ;;
        *)
            echo "❌ Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"