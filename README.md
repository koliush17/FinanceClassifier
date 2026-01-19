# Transaction Purpose Classification

This project is a text classification solution to predict the type of a financial transaction based on its purpose text. It includes a complete machine learning pipeline, from data preprocessing and model training to a REST API for serving the best-performing model.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Running the API Server](#running-the-api-server)
  - [Making API Requests](#making-api-requests)
- [Testing](#testing)

## Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/koliush17/FinanceClassifier   
    cd FinanceClassifier
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management. Make sure you have it installed.
    ```sh
    # Install project dependencies
    uv pip install -e .
    ```
## Usage

### Data Preprocessing

The data preprocessing steps are documented in `src/mlproject/data/Data_Processing.md`. The preprocessing of the text data is handled automatically as part of the training pipeline by the `TfidfVectorizer` in the `Classifier` class.

### Model Training and Evaluation

To run the full training pipeline, which includes model comparison, hyperparameter tuning, and saving the best model, run the following command(Should be done before running docker because it trains and saves a model which is used for classification):

```sh
uv run -m mlproject.run_pipeline
```

The training process and evaluation metrics are documented in `src/mlproject/models/classic/Training.md`.

### Running the API Server

To start the FastAPI server use Docker or run the following command:

```sh
uv run uvicorn src.mlproject.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Docker

You can also build and run the application using Docker.

1.  **Build the Docker image:**
    ```sh
    docker build -t transaction-classifier .
    ```

2.  **Run the Docker container:**
    ```sh
    docker run -p 8000:8000 transaction-classifier
    ```
The API will be available at `http://localhost:8000/docs`.

## Testing

To run the unit tests, use the following command:

```sh
uv run pytest
```
