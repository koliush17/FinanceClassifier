from importlib.resources import files

import pandas as pd 

def get_data() -> pd.DataFrame:
    data_path = files("mlproject.data") / "findata_processed.csv"
    df = pd.read_csv(data_path)

    return df 



