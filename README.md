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
- [Examples](#examples)

## Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/koliush17/FinanceClassifier   
    cd FinanceClassifier
    ```
2. **Create virtual environment**
    ```sh
    uv venv
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

Link to dataset: 
https://huggingface.co/datasets/mitulshah/transaction-categorization/viewer/default/train?p=45010&row=4501040

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
Or use curl directly in the terminal: 

```sh
curl -X 'POST'   'http://127.0.0.1:8000/classify'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
"purpose_text": "Your query"
}'
```

## Testing

To run the unit tests, use the following command:

```sh
uv run pytest
```

## Example

1. Example 1:
```sh
curl -X 'POST'   'http://127.0.0.1:8000/classify'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "purpose_text": "Starlink"
}'
```
Answer: {"transaction_type": "Utilities & Services"}

2. Example 2:
```sh
curl -X 'POST'   'http://127.0.0.1:8000/classify'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "purpose_text": "Wage"
}'
```
Answer: {"transaction_type": "Income"}

3. Example 3:
```sh
curl -X 'POST'   'http://127.0.0.1:8000/classify'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "purpose_text": "ATM Widthdrawal"
}'
```
Answer: {"transaction_type": "Financial Services"}
