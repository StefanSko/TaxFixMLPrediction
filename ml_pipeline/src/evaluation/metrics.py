"""
Metrics module for the ML pipeline.
This module provides functionality to evaluate classification models
and generate performance metrics.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate standard classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing precision, recall, F1-score, and accuracy
    """
    # Handle edge case of empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }

    # Calculate metrics
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }

    return metrics


def calculate_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate the ROC AUC score.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities for the positive class

    Returns:
        ROC AUC score
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_proba) == 0:
        return 0.0

    # Check if we have both positive and negative classes
    if len(np.unique(y_true)) < 2:
        return 0.0

    return roc_auc_score(y_true, y_proba)


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Generate a confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as a numpy array
    """
    # Handle edge case of empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.zeros((2, 2))

    return confusion_matrix(y_true, y_pred)


def evaluate_model(
        model,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained classification model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing all evaluation metrics and results
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    classification_metrics = calculate_classification_metrics(y_test, y_pred)

    try:
        roc_auc = calculate_roc_auc(y_test, y_proba)
    except Exception as e:
        roc_auc = 0.0
        print(f"Warning: Could not calculate ROC AUC: {e}")

    cm = generate_confusion_matrix(y_test, y_pred)

    # Compile all results
    results = {
        'metrics': classification_metrics,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'sample_count': len(y_test)
    }

    return results


def save_evaluation_results(results: Dict[str, Any], path: str) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results: Dictionary containing evaluation results
        path: Path where the results should be saved
    """
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value

    # Write to file
    with open(path, 'w') as f:
        json.dump(serializable_results, f, indent=4)


def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        output_path: Optional[str] = None
) -> None:
    """
    Plot the ROC curve.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities for the positive class
        output_path: Optional path to save the plot
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_proba) == 0 or len(np.unique(y_true)) < 2:
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Insufficient data)')
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        return

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    if output_path:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
        cm: np.ndarray,
        output_path: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix as a numpy array
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels
    classes = ['Negative (0)', 'Positive (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if output_path:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()