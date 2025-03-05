#!/usr/bin/env python
"""
Command-line interface for training ML models.

This script provides a command-line interface for training machine learning models
using the ModelTrainingPipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from training.pipeline import train_model_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a machine learning model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the training configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save the training results JSON file'
    )
    return parser.parse_args()


def main():
    """Main entry point for the training script."""
    args = parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        # Train model
        logger.info(f"Training model with configuration from {config_path}")
        results = train_model_from_config(str(config_path))

        # Output results
        print("\n=== Training Results ===")
        print(f"Model saved to: {results['model_path']}")
        print("\nEvaluation Metrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")

        # Save results if output path is provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()