## Training pipeline explanation & results

#### Pipeline Architecture

Training pipeline:
    1. Data Partitioning: The dataset was split into a training set and a test set.
    2. Model Selection: Four candidate algorithms (Logistic Regression, Random Forest, Decision Tree, and SVM) were evaluated using Cross-Validation on the training set.
    3. Hyperparameter Optimization: The best-performing model (SVM) underwent a hyperparameter search using RandomizedSearchCV with a pipeline to find the optimal configuration.
    4. Final Evaluation: The tuned model was evaluated on the test set to produce the final metrics below

#### Metrics

                           precision    recall  f1-score   support

       Charity & Donations       1.00      1.00      1.00        79
Entertainment & Recreation       0.99      1.00      0.99        77
        Financial Services       1.00      1.00      1.00        78
             Food & Dining       1.00      0.97      0.98        86
        Government & Legal       1.00      1.00      1.00        78
      Healthcare & Medical       0.94      1.00      0.97        84
                    Income       1.00      1.00      1.00        81
         Shopping & Retail       0.96      0.92      0.94        77
            Transportation       1.00      0.99      0.99        74
      Utilities & Services       0.99      1.00      0.99        86

                  accuracy                           0.99       800
                 macro avg       0.99      0.99      0.99       800
              weighted avg       0.99      0.99      0.99       800

High F1-scores, backed by consistent Precision and Recall, confirm successful model training and generalization

## Logs & process:

[20:10:05] [INFO] LogisticRegression f1 mean score: 0.9769
[20:10:08] [INFO] RandomForest f1 mean score: 0.9796
[20:10:08] [INFO] DecisionTree f1 mean score: 0.9759
[20:10:10] [INFO] SVM f1 mean score: 0.9830
[20:10:10] [INFO] Starting hyperparametr search for SVC()
[20:10:16] [INFO] Successfully found best parameters!
[20:10:16] [INFO] Best Parameters: {'clf__kernel': 'linear', 'clf__gamma': 'auto', 'clf__C': 10}
[20:10:16] [INFO] Best Cross-Validation Score: 0.9837478267727832
[20:10:16] [INFO] Successfully trained a SVC!
[20:10:16] [INFO] Generation report for SVC
 
Model cross-validation -> top performer model hyperparameter tuning -> evaluation
