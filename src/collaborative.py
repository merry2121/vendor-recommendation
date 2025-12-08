import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_scores(df, product_index, top_k=10):
    """
    Item-based collaborative filtering using product interactions.
    Uses historical sales as implicit feedback.
    """
    # Create matrix: vendor_id x product_id
    pivot = df.pivot_table(
        index="vendor_id",
        columns="product_id",
        values="historical_sales",
        fill_value=0
    )

    pivot_matrix = pivot.values

    try:
        # find column index of product in pivot
        prod_id = df.iloc[product_index]["product_id"]
        col_index = list(pivot.columns).index(prod_id)
    except ValueError:
        return [], []

    # cosine similarity between product columns
    similarities = cosine_similarity(pivot_matrix.T)

    similarity_scores = similarities[col_index]

    similar_items = similarity_scores.argsort()[::-1]

    # remove itself
    similar_items = similar_items[similar_items != col_index]

    return similar_items[:top_k], similarity_scores[similar_items][:top_k]
