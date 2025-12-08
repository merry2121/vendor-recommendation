import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def load_data(path="../data/synthetic_vendor_data.csv"):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    Preprocess the dataset:
    - One-hot encode categories
    - Normalize numeric values
    """
    numeric_cols = ["price", "historical_sales", "ctr", "add_to_cart"]
    
    # Scale numeric columns
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_cols])

    # One-hot encode categories
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cat = encoder.fit_transform(df[["category"]])

    # Build final feature matrix
    import numpy as np
    feature_matrix = np.concatenate([scaled_numeric, encoded_cat], axis=1)

    return feature_matrix, scaler, encoder
