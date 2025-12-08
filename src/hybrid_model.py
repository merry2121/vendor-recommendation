from content_based import content_based_scores
from collaborative import collaborative_scores
import numpy as np

def hybrid_recommend(df, feature_matrix, product_index, weight_cb=0.5, weight_cf=0.5, top_k=10):
    """
    Hybrid model = content-based + collaborative filtering.
    """
    cb_idx, cb_scores = content_based_scores(feature_matrix, product_index, top_k)
    cf_idx, cf_scores = collaborative_scores(df, product_index, top_k)

    # ensure arrays have same size
    min_len = min(len(cb_scores), len(cf_scores))

    cb_scores = cb_scores[:min_len]
    cf_scores = cf_scores[:min_len]

    # combine using weighted sum
    hybrid_scores = weight_cb * cb_scores + weight_cf * cf_scores

    # pick top recommendations
    ranked = hybrid_scores.argsort()[::-1]

    return ranked, hybrid_scores[ranked]
