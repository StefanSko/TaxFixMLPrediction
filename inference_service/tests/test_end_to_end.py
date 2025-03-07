#!/usr/bin/env python
"""
End-to-end integration test for the ML pipeline and inference service.

This script tests the full workflow from model training to inference service deployment.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import requests
from requests.exceptions import RequestException


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run end-to-end integration test")

    parser.add_argument(
        "--model-path",
        type=str,
        default="output/models/tax_filing_classifier.joblib",
        help="Path to the trained model file"
    )

    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default="output/models/preprocessor.joblib",
        help="Path to the preprocessor file"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for the inference service"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the inference service"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for service startup"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
        default=True
    )

    return parser.parse_args()


def start_inference_service(
        model_path: str,
        preprocessor_path: str,
        host: str = "localhost",
        port: int = 8000,
        debug: bool = False
) -> subprocess.Popen:
    """
    Start the inference service in a separate process.

    Args:
        model_path: Path to the trained model file
        preprocessor_path: Path to the preprocessor file
        host: Host for the inference service
        port: Port for the inference service
        debug: Whether to enable debug output

    Returns:
        Process object for the inference service
    """
    print(f"Starting inference service with model: {model_path}")
    print(f"Preprocessor: {preprocessor_path}")

    # Ensure the model and preprocessor files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

    # Prepare environment variables for the service
    env = os.environ.copy()
    env.update({
        "MODEL_PATH": model_path,
        "PREPROCESSOR_PATH": preprocessor_path,
        "API_HOST": host,
        "API_PORT": str(port),
        "API_DEBUG": "true" if debug else "false",
        # Generate a test API key
        "API_KEY": "test-api-key-for-integration-testing"
    })

    # Start the inference service
    cmd = [
        sys.executable,
        "-m",
        "inference_service.api.server",
    ]

    # Use subprocess.PIPE for stdout/stderr if not in debug mode
    stdout = None if debug else subprocess.PIPE
    stderr = None if debug else subprocess.PIPE

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr
    )

    print(f"Inference service started with PID: {process.pid}")
    return process


def wait_for_service(
        base_url: str,
        timeout: int = 30,
        check_interval: float = 0.5
) -> bool:
    """
    Wait for the inference service to become available.

    Args:
        base_url: Base URL of the inference service
        timeout: Maximum time to wait in seconds
        check_interval: Interval between checks in seconds

    Returns:
        True if the service is available, False otherwise
    """
    print(f"Waiting for inference service to become available at {base_url}...")

    health_url = f"{base_url}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print("Inference service is healthy!")
                    return True
                else:
                    print(f"Service status: {data.get('status')}")
        except RequestException:
            pass

        time.sleep(check_interval)

    print(f"Timed out waiting for inference service after {timeout} seconds")
    return False


def generate_test_data() -> List[Dict[str, Any]]:
    """
    Generate test data for prediction requests.

    Returns:
        List of test data dictionaries
    """
    # Define possible values for categorical features
    employment_types = ["employed", "self_employed", "unemployed", "retired", "student"]
    marital_statuses = ["single", "married", "divorced", "widowed", "separated"]
    device_types = ["mobile", "desktop", "tablet"]
    referral_sources = ["search", "social", "email", "friend", "advertisement", "other"]

    # Generate a list of test cases
    test_data = []

    # Generate 5 test cases
    for _ in range(5):
        data = {
            "age": random.randint(18, 80),
            "income": random.uniform(10000, 200000),
            "employment_type": random.choice(employment_types),
            "marital_status": random.choice(marital_statuses),
            "time_spent_on_platform": random.uniform(1, 120),
            "number_of_sessions": random.randint(1, 10),
            "fields_filled_percentage": random.uniform(10, 100),
            "previous_year_filing": random.choice([True, False]),
            "device_type": random.choice(device_types),
            "referral_source": random.choice(referral_sources)
        }
        test_data.append(data)

    return test_data


def test_prediction_endpoint(
        base_url: str,
        api_key: str,
        test_data: Optional[List[Dict[str, Any]]] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Test the prediction endpoint.

    Args:
        base_url: Base URL of the inference service
        api_key: API key for authentication
        test_data: Optional list of test data dictionaries

    Returns:
        Tuple of (success flag, list of results)
    """
    print("Testing prediction endpoint...")

    if test_data is None:
        test_data = generate_test_data()

    prediction_url = f"{base_url}/api/v1/predict"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }

    results = []
    success = True

    for i, data in enumerate(test_data):
        print(f"Sending test request {i + 1}/{len(test_data)}...")

        try:
            response = requests.post(
                prediction_url,
                headers=headers,
                json=data,
                timeout=10
            )

            result = {
                "request": data,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }

            if response.status_code != 200:
                print(f"Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                success = False
            else:
                print(f"Request succeeded: {json.dumps(response.json(), indent=2)}")

            results.append(result)

        except RequestException as e:
            print(f"Request error: {str(e)}")
            results.append({
                "request": data,
                "error": str(e),
                "success": False
            })
            success = False

    return success, results


def run_end_to_end_test(args: argparse.Namespace) -> int:
    """
    Run the full end-to-end test.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    print("Starting end-to-end integration test...")

    # Resolve paths
    model_path = os.path.abspath(args.model_path)
    preprocessor_path = os.path.abspath(args.preprocessor_path)

    # Start the inference service
    process = None
    try:
        # Start the inference service
        process = start_inference_service(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            host=args.host,
            port=args.port,
            debug=args.debug
        )

        # Wait for the service to become available
        base_url = f"http://{args.host}:{args.port}"
        if not wait_for_service(base_url, timeout=args.timeout):
            print("Failed to start inference service")
            return 1

        # Test the prediction endpoint
        api_key = "test-api-key-for-integration-testing"
        success, results = test_prediction_endpoint(base_url, api_key)

        # Print summary
        print("\nTest Summary:")
        print(f"Total requests: {len(results)}")
        print(f"Successful requests: {sum(1 for r in results if r.get('success', False))}")
        print(f"Failed requests: {sum(1 for r in results if not r.get('success', False))}")

        # Save results to file
        output_dir = Path("output/integration_tests")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"integration_test_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "success": success,
                "results": results
            }, f, indent=2)

        print(f"Results saved to: {output_file}")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("Test interrupted by user")
        return 130
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return 1
    finally:
        # Clean up resources
        if process is not None:
            print(f"Stopping inference service (PID: {process.pid})...")
            try:
                # Try to terminate gracefully first
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                process.kill()
                process.wait()
            print("Inference service stopped")


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(run_end_to_end_test(args))