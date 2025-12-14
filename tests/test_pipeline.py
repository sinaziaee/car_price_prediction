# Simple tests to validate the ML pipeline
import pytest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_input_json_format():
    """Test that input.json has the correct format."""
    with open("input.json", "r") as f:
        data = json.load(f)
    
    required_fields = ["miles", "year"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
        
    assert isinstance(data["miles"], (int, float)), "Miles should be numeric"
    assert isinstance(data["year"], (int, float)), "Year should be numeric"

def test_processor_import():
    """Test that the processor module can be imported."""
    try:
        from processor import Processor
        assert Processor is not None
    except ImportError as e:
        pytest.fail(f"Cannot import Processor: {e}")

def test_bentoml_service_import():
    """Test that BentoML service can be imported (when models exist)."""
    try:
        # This might fail if models don't exist, which is expected in CI
        import bentoml_service
        assert hasattr(bentoml_service, 'CarPricePrediction')
    except (ImportError, FileNotFoundError):
        # Expected when running in CI without trained models
        pytest.skip("Models not available for testing")

def test_main_module_import():
    """Test that main module can be imported."""
    import main
    assert hasattr(main, 'preprocess')
    assert hasattr(main, 'train_eval')
    assert hasattr(main, 'infer')

if __name__ == "__main__":
    pytest.main([__file__])