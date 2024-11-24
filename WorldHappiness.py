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

happiness_df = pd.read_csv("WorldHappiness/2019.csv")

drop_features = ["Overall rank"]
target_column = "Score"

train_df, test_df = train_test_split(happiness_df, test_size=0.2, random_state=42)

X_train = train_df.drop(columns=drop_features + [target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=drop_features + [target_column])
y_test = test_df[target_column]

numerical_features = X_train.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

linear_reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)


cv_folds = 6

# Before Cross-Validation
# Train the pipeline on the training set
linear_reg_pipeline.fit(X_train, y_train)

# Compute training and testing scores (before CV)
train_score_before = linear_reg_pipeline.score(X_train, y_train)
test_score_before = linear_reg_pipeline.score(X_test, y_test)

print("Before Cross-Validation:")
print("Training Score (R^2):", train_score_before)
print("Testing Score (R^2):", test_score_before)

# Perform cross-validation on the training set
cv_scores = cross_val_score(
    estimator=linear_reg_pipeline,
    X=X_train,
    y=y_train,
    cv=cv_folds,  # 5-fold cross-validation
    scoring="r2",  # Use R^2 score
)

# Compute the mean cross-validation score (as "test score after CV")
mean_cv_score = np.mean(cv_scores)

# Refit the pipeline on the entire training set to calculate training score
linear_reg_pipeline.fit(X_train, y_train)
train_score_after = linear_reg_pipeline.score(X_train, y_train)

print("\nAfter Cross-Validation:")
print("Training Score (R^2):", train_score_after)
print("Mean Cross-Validation Testing Score (R^2):", mean_cv_score)

print("-----------------------------------------")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),  # Reuse the same preprocessor as linear regression
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Compute training and testing scores for Random Forest
rf_train_score = rf_pipeline.score(X_train, y_train)
rf_test_score = rf_pipeline.score(X_test, y_test)

print("\nRandom Forest Performance:")
print("Training Score (R^2):", rf_train_score)
print("Testing Score (R^2):", rf_test_score)

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv_folds, scoring="r2")
rf_mean_cv_score = np.mean(rf_cv_scores)

print("\nRandom Forest Cross-Validation Mean Score (R^2):", rf_mean_cv_score)