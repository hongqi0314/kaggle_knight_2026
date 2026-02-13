# Kaggle Knight 2026 - Insurance Premium Prediction

## Competition

This project is our solution for the [Kaggle Playground Series - Season 4, Episode 12 (S4E12)](https://www.kaggle.com/competitions/playground-series-s4e12): **Regression with an Insurance Dataset**.

The training and test datasets are synthetically generated from a deep learning model trained on a real-world [Insurance Premium Prediction](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction) dataset. Feature distributions are close to, but not exactly the same as, the original.

## Business Problem

Insurance companies need to accurately estimate the premium amount to charge each policyholder. Setting premiums too high drives customers away; setting them too low leads to financial losses. The goal is to build a regression model that predicts the **Premium Amount** for a given policyholder based on their demographic, financial, health, and policy attributes.

## Dataset

| Split | Rows |
|-------|------|
| Train | 1,200,000 |
| Test  | 800,000 |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Age of the policyholder |
| Gender | Categorical | Male / Female |
| Annual Income | Numeric | Yearly income |
| Marital Status | Categorical | Married / Single / Divorced |
| Number of Dependents | Numeric | Count of dependents |
| Education Level | Categorical | High School / Bachelor's / Master's / PhD |
| Occupation | Categorical | Employed / Self-Employed / (missing) |
| Health Score | Numeric | Continuous health indicator |
| Location | Categorical | Urban / Suburban / Rural |
| Policy Type | Categorical | Basic / Comprehensive / Premium |
| Previous Claims | Numeric | Number of prior insurance claims |
| Vehicle Age | Numeric | Age of insured vehicle (years) |
| Credit Score | Numeric | Creditworthiness score |
| Insurance Duration | Numeric | Policy length (years) |
| Policy Start Date | Datetime | When the policy began |
| Customer Feedback | Categorical | Poor / Average / Good |
| Smoking Status | Categorical | Yes / No |
| Exercise Frequency | Categorical | Daily / Weekly / Monthly / Rarely |
| Property Type | Categorical | House / Apartment / Condo |

### Target

**Premium Amount** -- the annual insurance premium (continuous, regression target).

## Evaluation

Submissions are scored on **RMSLE (Root Mean Squared Logarithmic Error)**:

$$\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }$$

RMSLE penalizes under-predictions more than over-predictions and is scale-invariant, focusing on relative error rather than absolute differences.

## Project Structure

```
├── data/
│   ├── raw/            # Original competition CSVs (git-ignored)
│   └── processed/      # Engineered features & folds
├── notebooks/          # EDA and experimentation (Marimo .py notebooks)
├── src/                # Reusable modules (preprocessing, models, etc.)
├── EXPERIMENTS.md      # Experiment log (model, CV, LB, params)
├── CLAUDE.md           # Dev workflow & conventions
└── pyproject.toml      # Dependencies managed with uv
```

## Tech Stack

- **Python 3.12+** with [uv](https://github.com/astral-sh/uv) for environment management
- **Polars** for fast, memory-efficient data processing
- **scikit-learn** for preprocessing and evaluation
- **XGBoost / LightGBM** for gradient-boosted tree models
- **Marimo** for interactive notebook development
