# WorldHappiness

## Happiness Score Prediction

This repository contains a machine learning project focused on predicting happiness scores using the [2019 World Happiness Report dataset from Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness/data). The project explores data preprocessing, model evaluation, and a comparison of regression techniques, including Linear Regression and Random Forest Regressor.

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

### Insights
Random Forest performs better overall with the cross-validation mean score (𝑅^2 = 0.7842) being the highest score.
