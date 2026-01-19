param_grids = {
    "LogisticRegression": {
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["liblinear"],
        "clf__max_iter": [100, 500, 1000],
    },

    "RandomForest": {
        "clf__n_estimators": [100, 200, 500],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    },

    "DecisionTree": {
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
    },

    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    },
}

