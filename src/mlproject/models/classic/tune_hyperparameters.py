from mlproject.models.classic.classifier import Classifier
from mlproject.interfaces.sklearn_protocol import SklearnClassifier
from mlproject.utils.logger import get_logger

from sklearn.model_selection import RandomizedSearchCV

from typing import Dict

class HyperparameterTuner:
    def __init__(self,
                 model: SklearnClassifier,
                 parameters: Dict) -> None:

        self.model = model 
        self.parameters = parameters
        
        self.logger = get_logger("main")

    def fit(self,
            X: str,
            y: str,
            n_iter: int = 100,
            cv: int = 5,
            scoring: str = "f1_macro",
            verbose: int = 0):

        """Perform hyperparamters tuning"""

        self.logger.info(f"Starting hyperparameter search for {self.model}")

        clf = Classifier(model=self.model)
        pipeline = clf.pipeline

        randomized_search = RandomizedSearchCV(
            pipeline,
            param_distributions=self.parameters,
            n_iter=n_iter, 
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=verbose
        )

        randomized_search.fit(X, y)

        self.logger.info("Successfully found best parameters!")
        self.logger.info(f"Best Parameters: {randomized_search.best_params_}")
        self.logger.info(f"Best Cross-Validation Score: {randomized_search.best_score_}")

        clean_params = {k.replace('clf__', ''): v for k, v in randomized_search.best_params_.items()}

        return clean_params


