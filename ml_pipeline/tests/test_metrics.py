"""
Tests for the metrics module.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import json

from evaluation.metrics import (
    calculate_classification_metrics,
    calculate_roc_auc,
    generate_confusion_matrix,
    evaluate_model,
    save_evaluation_results,
    plot_roc_curve,
    plot_confusion_matrix
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, predictions, probabilities):
        self.predictions = predictions
        self.probabilities = probabilities

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        # Return probabilities for both classes (0 and 1)
        return np.column_stack((1 - self.probabilities, self.probabilities))


class TestMetrics:

    def setup_method(self):
        """Set up test data."""
        self.y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])
        self.y_proba = np.array([0.2, 0.9, 0.6, 0.8, 0.3, 0.1, 0.7, 0.6])

        # Create a mock model
        self.model = MockModel(self.y_pred, self.y_proba)

        # Create test dataframes
        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(8),
            'feature2': np.random.rand(8)
        })
        self.y_test_series = pd.Series(self.y_true)

    def test_calculate_classification_metrics(self):
        """Test calculation of classification metrics."""
        metrics = calculate_classification_metrics(self.y_true, self.y_pred)

        assert isinstance(metrics, dict)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'accuracy' in metrics

        # Test with empty arrays
        empty_metrics = calculate_classification_metrics(np.array([]), np.array([]))
        assert empty_metrics['precision'] == 0.0
        assert empty_metrics['recall'] == 0.0

    def test_calculate_roc_auc(self):
        """Test calculation of ROC AUC score."""
        roc_auc = calculate_roc_auc(self.y_true, self.y_proba)

        assert isinstance(roc_auc, float)
        assert 0.0 <= roc_auc <= 1.0

        # Test with empty arrays
        empty_roc_auc = calculate_roc_auc(np.array([]), np.array([]))
        assert empty_roc_auc == 0.0

        # Test with single class
        single_class_roc_auc = calculate_roc_auc(np.ones(5), np.random.rand(5))
        assert single_class_roc_auc == 0.0

    def test_generate_confusion_matrix(self):
        """Test generation of confusion matrix."""
        cm = generate_confusion_matrix(self.y_true, self.y_pred)

        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)

        # Test with empty arrays
        empty_cm = generate_confusion_matrix(np.array([]), np.array([]))
        assert empty_cm.shape == (2, 2)
        assert np.all(empty_cm == 0)

    def test_evaluate_model(self):
        """Test model evaluation."""
        # Test with numpy arrays
        results = evaluate_model(self.model, self.X_test.values, self.y_true)

        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'roc_auc' in results
        assert 'confusion_matrix' in results
        assert 'sample_count' in results

        # Test with pandas DataFrame and Series
        results_pd = evaluate_model(self.model, self.X_test, self.y_test_series)

        assert isinstance(results_pd, dict)
        assert results_pd['sample_count'] == len(self.y_test_series)

    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        results = {
            'metrics': {
                'precision': 0.75,
                'recall': 0.6,
                'f1_score': 0.67,
                'accuracy': 0.7
            },
            'roc_auc': 0.8,
            'confusion_matrix': np.array([[3, 1], [2, 4]]).tolist(),
            'sample_count': 10
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results', 'eval_results.json')
            save_evaluation_results(results, output_path)

            # Check if file exists
            assert os.path.exists(output_path)

            # Check if content is correct
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)

            assert loaded_results['metrics']['precision'] == 0.75
            assert loaded_results['roc_auc'] == 0.8
            assert loaded_results['sample_count'] == 10

    def test_plot_functions(self):
        """Test plotting functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test ROC curve plotting
            roc_path = os.path.join(tmpdir, 'roc_curve.png')
            plot_roc_curve(self.y_true, self.y_proba, roc_path)
            assert os.path.exists(roc_path)

            # Test confusion matrix plotting
            cm = generate_confusion_matrix(self.y_true, self.y_pred)
            cm_path = os.path.join(tmpdir, 'confusion_matrix.png')
            plot_confusion_matrix(cm, cm_path)
            assert os.path.exists(cm_path)

            # Test with edge cases
            edge_roc_path = os.path.join(tmpdir, 'edge_roc.png')
            plot_roc_curve(np.ones(5), np.random.rand(5), edge_roc_path)
            assert os.path.exists(edge_roc_path)