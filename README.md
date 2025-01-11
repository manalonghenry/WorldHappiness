# WorldHappiness

## Happiness Score Prediction

This repository contains a machine learning project focused on predicting happiness scores using the [2019 World Happiness Report dataset from Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness/data). The project explores data preprocessing, model evaluation, and a comparison of regression techniques, including Linear Regression, Random Forest Regressor and XGBoost.

### Dataset

The dataset used in this project is the 2019 World Happiness Report available on Kaggle.
Key features in the dataset include:

- GDP per Capita: Economic prosperity of a country.
- Social Support: Perceived social network support.
- Healthy Life Expectancy: Life expectancy adjusted for health.
- Freedom to Make Life Choices: Individual autonomy.
- Generosity: Charitable donations and generosity.
- Perceptions of Corruption: Trust in governance.
- Target Variable: Score — The happiness score of a country.
- Dropped Feature: Overall Rank.

### Installation

Clone the repository via git clone and install the required Python libraries via:

```
pip install -r requirements.txt
```

### Results

| Metric                      | Linear Regression | Random Forest | XGBoost  |
|-----------------------------|-------------------|---------------|----------|
| Training Score (without CV) | 1.000000          | 0.972398      | 0.999999 |
| Testing Score (without CV)  | 0.601909          | 0.648493      | 0.650734 |
| Training Score (with CV)    | 1.000000          | 0.972398      | 0.999999 |
| Testing Score (with CV)     | 0.755904          | 0.784200      | 0.794331 |

#### Conclusion
- XGBoost was the best-performing model in this context with and without cross validation.
- Linear Regression was the least effective model for this dataset most likely because it assumes a linear relationship which does not capture complex dynamics.
