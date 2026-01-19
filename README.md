# Vendor Recommendation System

Project Overview
A data-driven hybrid recommendation system designed for Zemen Bazaar and MeMi PLC. It predicts vendor preferences using machine learning techniques, combining Collaborative Filtering (CF) and Content-Based Filtering (CBF) for accurate and personalized recommendations.

Dataset
- Contains ~1,000 vendor–product interactions.
- Features:

| Feature           | Description                                   |
|------------------|-----------------------------------------------|
| ProductID        | Unique identifier for each product            |
| VendorID         | Identifier for the seller or producer        |
| ProductCategory  | Product category                              |
| Price            | Product selling price                          |
| HistoricalSales  | Number of units sold historically             |
| StockLevel       | Current inventory availability                |
| CustomerRating   | Customer review analysis (positive/negative) |

- Stored in CSV format and preprocessed in Python.
- Synthetic augmentation and normalization applied for consistency.

Model
Collaborative Filtering (CF)
- Uses Matrix Factorization to find hidden relationships between vendors and products.
- Recommends items based on similar vendors’ successful interactions.

Content-Based Filtering (CBF)
- Uses product attributes to recommend items similar to previously successful products.
- Computes similarity using cosine similarity of feature vectors.

Hybrid Model
- Combines CF and CBF predictions using a weighted average:
\[
R_{hybrid} = \alpha \times R_{CF} + (1 - \alpha) \times R_{CBF}
\]
- Balances personalization and generalization, addressing cold-start and sparsity issues.

Project Structure

vendor-recommendation/
├─ data/ # CSV datasets
├─ notebooks/ # Jupyter notebooks for analysis
├─ models/ # Trained models
├─ src/ # Scripts for preprocessing, training, and recommendations
├─ README.md
└─ requirements.txt
