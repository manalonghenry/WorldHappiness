from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

# Load the dataset
happiness_df = pd.read_csv("WorldHappiness/2019.csv")

# Define the target and features
drop_features = ["Overall rank"]
target_column = "Score"

# Split the data into training and testing sets
train_df, test_df = train_test_split(happiness_df, test_size=0.2, random_state=42)
X_train = train_df.drop(columns=drop_features + [target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=drop_features + [target_column])
y_test = test_df[target_column]

# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(include=["float64", "int64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Initialize models
dummy_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="mean")),  # Using DummyRegressor
    ]
)

linear_reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=100, random_state=42)),
    ]
)

# Evaluate models
cv_folds = 6

def evaluate_pipeline(pipeline, name):
    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Scores without cross-validation
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="r2")
    mean_cv_score = np.mean(cv_scores)

    return {
        "Model": name,
        "Training Score (without CV)": train_score,
        "Testing Score (without CV)": test_score,
        "Mean Training Score (with CV)": pipeline.score(X_train, y_train),
        "Mean Testing Score (with CV)": mean_cv_score,
    }

# Collect results for all models
results = []
results.append(evaluate_pipeline(dummy_pipeline, "Dummy Regressor"))
results.append(evaluate_pipeline(linear_reg_pipeline, "Linear Regression"))
results.append(evaluate_pipeline(rf_pipeline, "Random Forest"))
results.append(evaluate_pipeline(xgb_pipeline, "XGBoost"))

# Convert results to DataFrame for display
results_df = pd.DataFrame(results)

# Print results
print("\nModel Performance Comparison:")
print(results_df)

