# Bangalore House Price Prediction

A complete end-to-end machine learning project that predicts residential 
property prices in Bangalore based on location, size, and other features.

## Project Status
Model training complete. Deployment (FastAPI + HTML frontend) in progress.

## Dataset
Kaggle - Bangalore House Price Dataset
~13,000 raw listings, cleaned down to ~10,800 usable records.

## What I Did

### Data Cleaning
- Handled inconsistent formats in `total_sqft` (ranges like "1000-1200" → averaged)
- Extracted BHK count from size column ("2 BHK" → 2)
- Removed properties with impossible sqft per BHK (below 300 or above 1000)
- Removed properties where bathrooms exceeded BHK count by more than 2
- Grouped 1298 rare locations into "other" category (threshold: 50 listings)
- Removed price outliers above 5 crore

### EDA
- Top 10 most expensive locations by average price per sqft
- BHK count vs price distribution
- Bathroom count vs price
- Area type impact on price per sqft
- Availability impact on price (found it irrelevant, dropped it)

### Feature Engineering
- Created `sqft_per_bhk` — total sqft divided by BHK count
- Applied **target encoding** for location — replaced location names with 
  mean price per location calculated from training data only (prevents leakage)
- One hot encoded `area_type`

### Model Training
Trained and compared 5 models:

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.55 |
| Ridge Regression | 0.55 |
| Random Forest | 0.66 |
| LightGBM | 0.68 |
| XGBoost (tuned) | **0.745** |

Hyperparameter tuning done using RandomizedSearchCV with 5-fold cross validation.

### Final Model
**XGBoost Regressor — R² 0.745, selected as final model**

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- Matplotlib, Seaborn
- FastAPI (deployment — in progress)

## Project Structure
House-Price-Prediction-System/
│
├── notebook/
│   ├── bangalore_house_price.ipynb
│   └── Bengaluru_House_Data.csv
│
├── app/
│   ├── model/
│   │   ├── model.pkl
│   │   └── price_per_location.pkl
│   ├── schema/
│       ├── user_input.py
│   ├── main.py
│   └── frontend.html
│
└── README.md

## Deployment
FastAPI backend with HTML frontend — currently in progress. 
Will update with live link once deployed.

## Key Learnings
- Target encoding outperformed one hot encoding for high cardinality location column
- Location is the strongest predictor of price in Bangalore real estate
- Tree based models significantly outperform linear models on this non-linear dataset
