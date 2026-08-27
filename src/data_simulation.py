"""
data_simulation.py
Creates the synthetic logistics dataset used in the project.
"""

import numpy as np
import pandas as pd


def create_dataset(n=500, random_state=42):
    np.random.seed(random_state)

    region = np.random.choice(["North", "South", "East", "West"], n)
    route_type = np.random.choice(
        ["Local", "Regional", "Intercity"], n, p=[0.42, 0.35, 0.23]
    )
    transport_mode = np.random.choice(
        ["Bike", "Van", "Truck"], n, p=[0.25, 0.45, 0.30]
    )
    weather = np.random.choice(
        ["Clear", "Cloudy", "Rain"], n, p=[0.60, 0.25, 0.15]
    )
    traffic = np.random.choice(
        ["Low", "Medium", "High"], n, p=[0.30, 0.45, 0.25]
    )

    distance = np.round(np.random.gamma(3.0, 30.0, n) + 5, 1)
    shipment_volume = np.random.poisson(40, n) + 5
    fuel_price = np.round(np.random.normal(96, 5, n), 2)

    route_effect = {"Local": 0.2, "Regional": 0.8, "Intercity": 1.4}
    mode_effect = {"Bike": 0.1, "Van": 0.35, "Truck": 0.65}
    weather_effect = {"Clear": 0.0, "Cloudy": 0.25, "Rain": 0.9}
    traffic_effect = {"Low": 0.0, "Medium": 0.55, "High": 1.35}

    delivery_time = (
        1.8
        + 0.035 * distance
        + 0.018 * shipment_volume
        + np.array([route_effect[x] for x in route_type])
        + np.array([mode_effect[x] for x in transport_mode])
        + np.array([weather_effect[x] for x in weather])
        + np.array([traffic_effect[x] for x in traffic])
        + np.random.normal(0, 0.65, n)
    )

    delivery_time = np.maximum(delivery_time, 1.0)
    expected_time = 2.2 + 0.028 * distance + 0.010 * shipment_volume
    delay = np.maximum(
        0, delivery_time - expected_time + np.random.normal(0, 0.35, n)
    )

    transport_cost = (
        75
        + 3.1 * distance
        + 0.75 * shipment_volume
        + 0.35 * fuel_price * (distance / 100)
        + np.array([
            10 if x == "Van" else 25 if x == "Truck" else 3
            for x in transport_mode
        ])
        + np.random.normal(0, 35, n)
    )
    transport_cost = np.maximum(transport_cost, 50)

    return pd.DataFrame({
        "Region": region,
        "Route_Type": route_type,
        "Transport_Mode": transport_mode,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Distance_km": distance,
        "Shipment_Volume": shipment_volume,
        "Fuel_Price_INR_L": fuel_price,
        "Delivery_Time_hr": np.round(delivery_time, 2),
        "Expected_Time_hr": np.round(expected_time, 2),
        "Delay_hr": np.round(delay, 2),
        "Transport_Cost_INR": np.round(transport_cost, 2)
    })


if __name__ == "__main__":
    data = create_dataset()
    data.to_csv("data/logistics_data.csv", index=False)
    print("Dataset created successfully.")
