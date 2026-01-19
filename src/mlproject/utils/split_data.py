from sklearn.model_selection import train_test_split

import pandas as pd 

def apply_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        df["purpose_text"],
        df["transaction_type"],
        test_size=test_size,
        stratify=df["transaction_type"],
        random_state=42
        )

    return (X_train, X_test, y_train, y_test)
