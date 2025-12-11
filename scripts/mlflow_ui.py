#!/usr/bin/env python3
"""
Helper script to launch MLflow UI with proper configuration.

Usage:
    python scripts/mlflow_ui.py [--port PORT]

This script:
1. Suppresses urllib3 SSL warnings (LibreSSL compatibility on macOS)
2. Automatically detects the correct database path
3. Launches MLflow UI with the correct backend-store-uri
"""

import os
import sys
import warnings
import argparse

# Suppress urllib3 SSL warning for LibreSSL compatibility (macOS)
# This must be done before any imports that load urllib3
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass  # urllib3 < 2.0 doesn't have NotOpenSSLWarning

# Also set environment variable for child processes
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::urllib3.exceptions.NotOpenSSLWarning"
)


def main():
    parser = argparse.ArgumentParser(
        description="Launch MLflow UI with Leap configuration"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run MLflow UI on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Check for mlflow.db in project root
    db_path = os.path.join(project_root, "mlflow.db")

    if not os.path.exists(db_path):
        print(f"Warning: MLflow database not found at {db_path}")
        print("Run a training session first to create the database.")
        print("Example: python main.py train --symbol EURUSD --epochs 1")
        print()

    tracking_uri = f"sqlite:///{db_path}"

    print(f"Starting MLflow UI...")
    print(f"  Database: {db_path}")
    print(f"  URL: http://{args.host}:{args.port}")
    print()

    # Import mlflow after warning suppression
    import mlflow.cli

    # Run mlflow ui
    sys.argv = [
        "mlflow", "ui",
        "--backend-store-uri", tracking_uri,
        "--host", args.host,
        "--port", str(args.port)
    ]

    mlflow.cli.cli()


if __name__ == "__main__":
    main()
