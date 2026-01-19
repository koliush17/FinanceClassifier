import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from mlproject.interfaces.sklearn_protocol import SklearnClassifier

class Classifier:
    def __init__(self, model: SklearnClassifier):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer='char_wb',
                stop_words=None, # remove 'the', 'a', etc.
                ngram_range=(3, 5), # looks for pairs like 'new shirt'
                min_df=1, # ignore words that appear only once in the whole dataset
                lowercase=True
            )),
            ("clf", model),
        ])

    def fit(self, X_train: pd.Series, y_train: pd.Series):
        """Train a model on training data"""

        self.pipeline.fit(X_train, y_train)

        return self # for method chaining. Pipeline expects it

    def score(self, X_test: pd.Series, y_test: pd.Series) -> float:
        """Calculate score of the trained model"""

        return self.pipeline.score(X_test, y_test)

    def predict(self, X: str)-> str:
        """Predict labels for input data"""

        return self.pipeline.predict([X])
