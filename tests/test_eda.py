import os
import sys
import pytest

# Add the project root to sys.path so we can import from data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.eda import perform_eda

def test_perform_eda(tmp_path):
    # tmp_path is a built-in pytest fixture that provides a temporary directory unique to the test invocation
    output_dir = str(tmp_path)
    
    # Run EDA on a tiny subsample of the dataset to ensure the test is fast
    results = perform_eda(output_dir=output_dir, split="train", subsample=50)
    
    # Check that plots were generated
    assert os.path.exists(os.path.join(output_dir, "class_distribution.png"))
    assert os.path.exists(os.path.join(output_dir, "length_distribution.png"))
    
    # Check that report was generated
    report_path = os.path.join(output_dir, "eda_report.md")
    assert os.path.exists(report_path)
    
    # Validate some content in the report
    with open(report_path, "r") as f:
        content = f.read()
        assert "GoEmotions Exploratory Data Analysis" in content
        assert "Class Imbalance" in content
        assert "Text Lengths" in content
        
    # Validate return dict
    assert "most_frequent_class" in results
    assert "least_frequent_class" in results
    assert results["total_samples"] == 50
    assert "mean_token_length" in results
    assert "percent_multi_label" in results
