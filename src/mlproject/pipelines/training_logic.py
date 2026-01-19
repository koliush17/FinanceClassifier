import pickle as pkl
import pandas as pd

from typing import Dict, Any, Tuple
from pathlib import Path

from mlproject.interfaces.sklearn_protocol import SklearnClassifier
from mlproject.models.classic.classifier import Classifier
from mlproject.utils.parameters_to_tune import param_grids 
from mlproject.models.classic.model_evaluation import BestModelFinder 
from mlproject.models.classic.tune_hyperparameters import HyperparameterTuner 
from mlproject.utils.logger import get_logger

logger = get_logger("main")

def find_best_model(models: Dict[str, SklearnClassifier], 
                    X_train: pd.Series, 
                    y_train: pd.Series) -> Tuple[SklearnClassifier, str]:

    """Find best model using cross validation"""

    find_model = BestModelFinder(models)
    best_model_name = find_model.fit(X_train, y_train)
    best_model = models[best_model_name]

    return best_model, best_model_name

def find_best_parameters(best_model: SklearnClassifier,
                         best_model_name: str,
                         X_train: pd.Series, 
                         y_train: pd.Series,
                         **kwargs) -> Dict[str, Any]:

    """Find best parameters using RandomSearch"""

    parameters = param_grids[best_model_name] 
    find_params = HyperparameterTuner(best_model, parameters)
    best_params = find_params.fit(X_train, y_train, **kwargs)

    return best_params

def train_best_model(best_model: SklearnClassifier, 
                     best_params: Dict[str, Any],
                     X_train: pd.Series,
                     y_train: pd.Series) -> Classifier:

    """Train the best performer model"""

    best_model.set_params(**best_params)
    clf = Classifier(best_model)
    model = clf.fit(X_train, y_train)

    return model

def save_model(model: Any, model_name: str, directory: str = "models/registry") -> None:
    """Saves the trained model to a specific directory.
       Returns the path where the model was saved"""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path / "model_latest.pkl"
    
    with open(file_path, "wb") as f:
        pkl.dump(model, f)
        
    logger.info(f"Model successfully saved to {file_path}")

