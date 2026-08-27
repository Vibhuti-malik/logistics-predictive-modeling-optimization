"""
model_training.py
Predicts delivery delay using Linear Regression and Random Forest.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "logistics_data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    cv_rmse = (-cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )).mean()

    print(f"\n{name}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R-squared: {r2:.3f}")
    print(f"5-fold CV RMSE: {cv_rmse:.3f}")

    return model, mae, rmse, r2, cv_rmse


def main():
    df = load_data()

    target = "Delay_hr"
    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = [
        "Region", "Route_Type", "Transport_Mode",
        "Weather", "Traffic_Level"
    ]
    numerical_features = [
        "Distance_km", "Shipment_Volume",
        "Fuel_Price_INR_L", "Delivery_Time_hr",
        "Expected_Time_hr", "Transport_Cost_INR"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    linear_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    random_forest = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    evaluate_model(
        "Linear Regression",
        linear_model, X_train, X_test, y_train, y_test
    )

    evaluate_model(
        "Random Forest",
        random_forest, X_train, X_test, y_train, y_test
    )


if __name__ == "__main__":
    main()
