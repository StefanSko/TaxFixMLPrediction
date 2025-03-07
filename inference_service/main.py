"""
Main entry point for the inference service.

This module provides the FastAPI application instance and serves as the entry point
for running the inference service.
"""

import argparse
import logging

from api.server import create_app, run_server

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run the inference service")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


# Create the FastAPI application
args = parse_args()
app = create_app(config_path=args.config)

# Get logger
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    run_server(config_path=args.config)