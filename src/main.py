from generate_dataset import *
from preprocessing import *
from hybrid_model import *

def run_recommender(product_index=10):
    df = load_data()
    features, scaler, encoder = preprocess_data(df)

    rec_indices, scores = hybrid_recommend(df, features, product_index)

    print("\nTop Recommendations:")
    for idx, score in zip(rec_indices[:5], scores[:5]):
        print(f"Product: {df.iloc[idx]['product_id']} | Score: {score:.4f}")

if __name__ == "__main__":
    run_recommender(10)
