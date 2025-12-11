FPL Player Performance Prediction

Machine learning tutorial demonstrating time-series regression for Fantasy Premier League points prediction with hyperparameter tuning and data leakage analysis.

Overview
This project compares three regression models (Linear Regression, Random Forest, XGBoost) for predicting FPL player performance, with emphasis on:
- Proper time-series validation
- Systematic hyperparameter tuning
- Data leakage detection and correction

Key Findings
- Initial models achieved R² > 0.97 due to data leakage from FPL-derived features (bonus, BPS, ICT metrics)
- Clean models with legitimate features achieved realistic R² ≈ 0.94
- Random Forest performed best with clean features (RMSE: 0.73 points)
- Demonstrates importance of feature validation in production ML systems

Installation
```bash
pip install -r requirements.txt
```

Usage

Open `fpl_prediction_tutorial.ipynb` in Jupyter/Colab and run all cells sequentially.

Data source:** FPL Official API (https://fantasy.premierleague.com/api/)

No authentication or external files required - all data fetched from API.

Project Structure
```
fpl-ml-prediction/
├── fpl_prediction_tutorial.ipynb  # Main tutorial notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

 Models Evaluated

1. Linear Regression
- Baseline model
- Best with leaky features (R² = 0.972)

2. Random Forest
- Optimal: n_estimators=200, max_depth=15
- Best with clean features (R² = 0.941)

3. XGBoost
- Optimal: max_depth=3, learning_rate=0.1
- Showed severe overfitting with default hyperparameters

Data Leakage Analysis

Problem: Features like `bonus` and `bps` are components of the FPL scoring formula:
```
total_points = base_points + bonus + adjustments
```

Solution: Removed FPL-derived features, retained only:
- Performance metrics (goals, assists, clean sheets)
- Expected stats (xG, xA)
- Defensive actions
- Match context

Result: Realistic R² ≈ 0.94 vs unrealistic R² ≈ 0.97

Key Techniques Demonstrated

- Time-series train/validation/test split (no look-ahead bias)
- Rolling average and lag feature engineering
- 5-method feature importance analysis
- Systematic hyperparameter tuning
- Data leakage detection and correction
- Model comparison and evaluation

Results Summary

| Model | Leaky R² | Clean R² | Clean RMSE |
|-------|----------|----------|------------|
| Linear Regression | 0.972 | 0.895 | 0.975 |
| Random Forest | 0.963 | 0.941 | 0.733 |
| XGBoost | 0.968 | 0.939 | 0.742 |


- Fantasy Premier League for API access
- Scikit-learn, XGBoost developers# fpl-ml-prediction
