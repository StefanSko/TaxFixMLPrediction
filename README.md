# Taxfix ML Prediction

## Project Overview

This project is an end-to-end machine learning solution to predict whether a user will complete their tax filing process. The system focuses on:

- Fast inference time for real-time predictions
- Well-structured architecture with clear separation of concerns
- Clear separation between training and inference components

## Architecture

The project is structured as a monorepo with two main packages:

1. **ML Pipeline**: Responsible for data processing, model training, and evaluation
2. **Inference Service**: Handles real-time prediction requests via a REST API

### ML Pipeline Components

- **Data**: Data loading, versioning, and storage utilities
- **Preprocessing**: Feature engineering, transformation, and normalization
- **Models**: Model definitions and configurations
- **Training**: Model training workflows and hyperparameter tuning
- **Evaluation**: Model evaluation metrics and performance analysis
- **Utils**: Shared utilities for the ML pipeline

### Inference Service Components

- **API**: FastAPI endpoints for prediction requests
- **Core**: Core business logic for the inference service
- **Services**: Internal services for model loading, caching, and prediction
- **Utils**: Shared utilities for the inference service

## Getting Started

### Prerequisites

- Python 3.12
- Poetry for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/taxfix-completion-prediction.git
cd taxfix-completion-prediction

# Install dependencies
poetry install
```

### Running the ML Pipeline

```bash
# Activate the virtual environment
poetry shell

# Run the ML pipeline
python -m ml_pipeline.training.train
```

### Running the Inference Service

```bash
# Activate the virtual environment
poetry shell

# Start the API server
uvicorn inference_service.api.main:app --reload
```

## Development

### Code Quality

This project uses:
- **Ruff** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run the quality checks:

```bash
# Run linting
poetry run ruff check .

# Run type checking
poetry run mypy .

# Run tests
poetry run pytest
```

