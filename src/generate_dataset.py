import pandas as pd
import numpy as np
import os

def generate_dataset(rows=1000, save_path="../data/synthetic_vendor_data.csv"):
    np.random.seed(42)

    categories = ["Electronics", "Clothing", "Home", "Beauty", "Sports"]

    data = pd.DataFrame({
        "vendor_id": np.random.randint(1, 51, rows),
        "product_id": np.random.randint(1000, 5000, rows),
        "category": np.random.choice(categories, rows),
        "price": np.random.uniform(10, 500, rows).round(2),
        "historical_sales": np.random.randint(0, 2000, rows),
        "ctr": np.random.uniform(0.01, 0.2, rows).round(3),
        "add_to_cart": np.random.randint(0, 300, rows),
    })

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_csv(save_path, index=False)

    print(f"Dataset saved to: {save_path}")
    print(data.head())


if __name__ == "__main__":
    generate_dataset()
