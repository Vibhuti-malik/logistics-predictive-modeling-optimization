# Logistics Predictive Modeling and Optimization

## Week 4 Task

This project demonstrates predictive modeling and optimization for a hypothetical logistics operation. The main objective is to forecast delivery delays and use the model results to support operational decisions.

## Project Objectives

- Define a logistics forecasting problem.
- Simulate a realistic logistics dataset.
- Explore important operational variables.
- Train and compare predictive models.
- Evaluate models using MAE, RMSE, and R-squared.
- Apply 5-fold cross-validation.
- Use model predictions to compare transport scenarios.
- Develop practical optimization recommendations.

## Dataset

The dataset contains 500 synthetic shipment records with variables including:

- Region
- Route type
- Transport mode
- Weather
- Traffic level
- Distance
- Shipment volume
- Fuel price
- Delivery time
- Expected delivery time
- Delay
- Transport cost

The dataset is synthetic and is intended for academic demonstration only.

## Models

Two regression models are implemented:

1. Linear Regression — used as a simple, interpretable baseline.
2. Random Forest Regressor — used to capture non-linear relationships between logistics variables.

The target variable is `Delay_hr`.

## Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared
- 5-fold cross-validation RMSE

## Optimization

`optimization.py` performs a scenario comparison for Bike, Van, and Truck transport modes. It predicts the expected delay for each option and applies a simple cost constraint to select a practical transport choice.

## Project Structure

```text
logistics-predictive-modeling-optimization/
│
├── data/
│   └── logistics_data.csv
│
├── src/
│   ├── data_simulation.py
│   ├── eda.py
│   ├── model_training.py
│   └── optimization.py
│
├── README.md
└── requirements.txt
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run exploratory analysis:

```bash
python src/eda.py
```

Run model training and evaluation:

```bash
python src/model_training.py
```

Run the optimization scenario:

```bash
python src/optimization.py
```

## Repository

https://github.com/Vibhuti-malik/logistics-predictive-modeling-optimization
