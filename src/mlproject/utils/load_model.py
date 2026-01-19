import pickle as pkl

from pathlib import Path

from mlproject.models.classic.classifier import Classifier
from mlproject.utils.logger import get_logger

logger = get_logger("main")

def load_model() -> Classifier:
    cwd = Path.cwd() # get parent path
    model_path = cwd / "models/registry/model_latest.pkl"

    logger.info(f"Loading a model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pkl.load(f)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    return model

