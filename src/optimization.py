"""
optimization.py
Simple scenario-based optimization using the trained Random Forest model.

The script compares transport choices for a shipment and selects the option
with the lowest predicted delay while keeping cost within a practical limit.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "logistics_data.csv"


def train_model(df):
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

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model


def compare_transport_modes(model):
    scenarios = []

    for mode in ["Bike", "Van", "Truck"]:
        scenarios.append({
            "Region": "North",
            "Route_Type": "Regional",
            "Transport_Mode": mode,
            "Weather": "Clear",
            "Traffic_Level": "Medium",
            "Distance_km": 120,
            "Shipment_Volume": 45,
            "Fuel_Price_INR_L": 96,
            "Delivery_Time_hr": 7.0,
            "Expected_Time_hr": 5.6,
            "Transport_Cost_INR": {
                "Bike": 475,
                "Van": 520,
                "Truck": 610
            }[mode]
        })

    scenario_df = pd.DataFrame(scenarios)
    scenario_df["Predicted_Delay_hr"] = model.predict(scenario_df)

    # Example decision rule: prioritize low delay, subject to a cost ceiling.
    feasible = scenario_df[scenario_df["Transport_Cost_INR"] <= 550]

    if len(feasible) > 0:
        best = feasible.sort_values("Predicted_Delay_hr").iloc[0]
    else:
        best = scenario_df.sort_values("Predicted_Delay_hr").iloc[0]

    print("\nTransport scenario comparison:")
    print(scenario_df[
        ["Transport_Mode", "Transport_Cost_INR", "Predicted_Delay_hr"]
    ].to_string(index=False))

    print(
        f"\nRecommended mode: {best['Transport_Mode']} "
        f"(predicted delay {best['Predicted_Delay_hr']:.2f} hr)"
    )


def main():
    df = pd.read_csv(DATA_PATH)
    model = train_model(df)
    compare_transport_modes(model)


if __name__ == "__main__":
    main()
