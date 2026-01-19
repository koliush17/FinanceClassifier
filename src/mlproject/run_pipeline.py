from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from mlproject.utils.read_data import get_data
from mlproject.utils.split_data import apply_train_test_split
from mlproject.models.classic.model_evaluation import evaluate_model
from mlproject.pipelines.training_logic import find_best_model, find_best_parameters, train_best_model, save_model

models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "SVM": SVC(),
    }

def run_training_pipeline() -> None:
    """Find best model and evaluate it"""

    df = get_data()
    X_train, X_test, y_train, y_test = apply_train_test_split(df)

    # Model layer
    best_model, best_model_name = find_best_model(models, X_train, y_train)

    # Parameters layer
    best_params = find_best_parameters(best_model, best_model_name, X_train, y_train)

    model = train_best_model(best_model, best_params, X_train, y_train)

    # Evaluation layer 
    metrics = evaluate_model(model.pipeline, X_test, y_test)
    
    # Model saving
    save_model(model, best_model_name)

    print(metrics["report"])

if __name__ == "__main__":
    run_training_pipeline()

