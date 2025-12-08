import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def content_based_scores(feature_matrix, product_index, top_k=10):
    """
    Computes content-based similarity using cosine similarity.
    Returns indices and similarity scores.
    """
    # similarity between selected product and all products
    similarities = cosine_similarity(
        [feature_matrix[product_index]], 
        feature_matrix
    )[0]

    # sort by highest similarity
    similar_indices = similarities.argsort()[::-1]

    # remove the product itself
    similar_indices = similar_indices[similar_indices != product_index]

    # return top-k recommended products
    return similar_indices[:top_k], similarities[similar_indices][:top_k]
