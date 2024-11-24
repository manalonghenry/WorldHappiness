from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

#Load
happiness_df = pd.read_csv("WorldHappiness/2019.csv")

drop_features = ["Overall rank"]
target_column = "Score"

#Train-test split
train_df, test_df = train_test_split(happiness_df, test_size=0.2, random_state=42)
X_train = train_df.drop(columns=drop_features + [target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=drop_features + [target_column])
y_test = test_df[target_column]

#Split features
numerical_features = X_train.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

#Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

#Linear Regression pipeline
linear_reg_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

#Random Forest pipeline
rf_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]
)

#XGBoost pipeline
xgb_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", XGBRegressor(n_estimators=100, random_state=42))]
)

#Cross-validation folds
cv_folds = 6

#Train then score Linear Regression
linear_reg_pipeline.fit(X_train, y_train)
train_score_before = linear_reg_pipeline.score(X_train, y_train)
test_score_before = linear_reg_pipeline.score(X_test, y_test)
cv_scores_lr = cross_val_score(linear_reg_pipeline, X_train, y_train, cv=cv_folds, scoring="r2")
mean_cv_score_lr = np.mean(cv_scores_lr)
train_score_after_lr = linear_reg_pipeline.score(X_train, y_train)

#Train then score Random Forest
rf_pipeline.fit(X_train, y_train)
rf_train_score = rf_pipeline.score(X_train, y_train)
rf_test_score = rf_pipeline.score(X_test, y_test)
cv_scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=cv_folds, scoring="r2")
rf_mean_cv_score = np.mean(cv_scores_rf)

#Train then score XGBoost
xgb_pipeline.fit(X_train, y_train)
xgb_train_score = xgb_pipeline.score(X_train, y_train)
xgb_test_score = xgb_pipeline.score(X_test, y_test)
cv_scores_xgb = cross_val_score(xgb_pipeline, X_train, y_train, cv=cv_folds, scoring="r2")
xgb_mean_cv_score = np.mean(cv_scores_xgb)

#Output results
results = {
    "Metric": [
        "Training Score (without CV)",
        "Testing Score (without CV)",
        "Training Score (with CV)",
        "Testing Score (with CV)"
    ],
    "Linear Regression": [
        train_score_before,  # Training score without cross-validation
        test_score_before,   # Testing score without cross-validation
        train_score_after_lr,    # Training score with cross-validation
        mean_cv_score_lr         # Testing score with cross-validation
    ],
    "Random Forest": [
        rf_train_score,       # Training score without cross-validation
        rf_test_score,        # Testing score without cross-validation
        rf_pipeline.score(X_train, y_train),  # Refit Random Forest for training with CV
        rf_mean_cv_score       # Testing score with cross-validation
    ],
    "XGBoost": [
        xgb_train_score,      # Training score without cross-validation
        xgb_test_score,       # Testing score without cross-validation
        xgb_pipeline.score(X_train, y_train),  # Refit XGBoost for training with CV
        xgb_mean_cv_score     # Testing score with cross-validation
    ]
}

# Create DataFrame for results
results_df = pd.DataFrame(results)

#Print results
print("\nModel Performance:")
print(results_df)