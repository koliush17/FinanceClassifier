import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from mlproject.main import app
from mlproject.utils.split_data import apply_train_test_split
from mlproject.models.classic.classifier import Classifier

# Test 1: Health check endpoint
def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test 2: apply_train_test_split function
def test_apply_train_test_split():
    data = {"text": ["a", "b", "c", "d", "e", "f"], "transaction_type": [1, 0, 1, 0, 2, 2]}
    df = pd.DataFrame(data)
    df.rename(columns={"text": "purpose_text"}, inplace=True)
    X_train, X_test, y_train, y_test = apply_train_test_split(df, test_size=0.5)
    assert len(X_train) == 3
    assert len(X_test) == 3

# Test 3: Classifier instantiation
def test_classifier_instantiation():
    model = Classifier(model=LogisticRegression())
    assert model is not None
    assert hasattr(model, "pipeline")

# Test 4: Classifier predict method type
def test_classifier_predict_type():
    classifier = Classifier(model=LogisticRegression())
    X = pd.Series(["test text 1", "test text 2"])
    y = pd.Series(["A", "B"])
    classifier.fit(X, y)
    prediction = classifier.predict("some text")
    assert isinstance(prediction[0], str)
