from mlproject.models.classic.classifier import Classifier
from mlproject.interfaces.sklearn_protocol import SklearnClassifier
from mlproject.utils.logger import get_logger

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.pipeline import Pipeline

from typing import List, Dict


def evaluate_model(pipeline: Pipeline, X_test: any, y_test: any) -> Dict:
    """Get metrics of the trained model"""
    
    y_pred = pipeline.predict(X_test)
    logger = get_logger("main")

    logger.info("Generation report")
    return {
        "accuracy": pipeline.score(X_test, y_test),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "report": classification_report(y_test, y_pred)
    }


class BestModelFinder:
    def __init__(self, models: List[SklearnClassifier]) -> None:
        self.models = models 
        self.logger = get_logger("main")

    def fit(self, X: str, y: str) -> str:
        """Perform model cross-validtion to find the best performing model"""

        results = {}

        for name, model in self.models.items():
            clf = Classifier(model=model)

            cv_scores = cross_val_score(clf.pipeline, X, y, cv=5, scoring="f1_macro")

            results[name] = cv_scores.mean(), 
            self.logger.info((f"{name} f1 mean score: {cv_scores.mean():.4f}"))

        return max(results, key=results.get)
