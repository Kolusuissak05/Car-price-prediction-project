import json
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def get_brand_name(car_name: str) -> str:
    return car_name.split(" ")[0].strip()


def clean_data(value: str) -> float:
    value = value.split(" ")[0].strip()
    if value == "":
        return 0.0
    return float(value)


def main():
    car_data = pd.read_csv("Cardetails.csv")

    # Drop torque (unused in the notebook's model) and clean rows
    car_data.drop(columns=["torque"], inplace=True)
    car_data.dropna(inplace=True)
    car_data.drop_duplicates(inplace=True)

    # Build a "Brand Model" grouping (first two words of the full name) for
    # the frontend's model dropdown + spec autofill, before we reduce the
    # name column down to just the brand for the actual regression features.
    car_data["model_short"] = car_data["name"].apply(
        lambda x: " ".join(x.split(" ")[:2]).strip()
    )
    car_data["brand_only"] = car_data["name"].apply(get_brand_name)

    models_by_brand = {}
    model_defaults = {}
    for brand, brand_df in car_data.groupby("brand_only"):
        model_names = sorted(brand_df["model_short"].unique().tolist())
        models_by_brand[brand] = model_names
        for model_name, model_df in brand_df.groupby("model_short"):
            model_defaults[model_name] = {
                "year": int(model_df["year"].median()),
                "fuel": model_df["fuel"].mode().iloc[0],
                "transmission": model_df["transmission"].mode().iloc[0],
                "mileage": round(float(model_df["mileage"].apply(clean_data).median()), 2),
                "engine": round(float(model_df["engine"].apply(clean_data).median()), 0),
                "max_power": round(float(model_df["max_power"].apply(clean_data).median()), 1),
                "seats": int(model_df["seats"].median()),
            }

    car_data.drop(columns=["model_short", "brand_only"], inplace=True)

    # Extract brand from full car name
    car_data["name"] = car_data["name"].apply(get_brand_name)

    brands = sorted(car_data["name"].unique().tolist())
    brand_to_code = {brand: i + 1 for i, brand in enumerate(brands)}
    car_data["name"] = car_data["name"].map(brand_to_code)

    # Clean numeric-with-unit columns
    car_data["mileage"] = car_data["mileage"].apply(clean_data)
    car_data["engine"] = car_data["engine"].apply(clean_data)
    car_data["max_power"] = car_data["max_power"].apply(clean_data)

    fuel_map = {"Diesel": 0, "Petrol": 1, "LPG": 2, "CNG": 3}
    seller_map = {"Individual": 0, "Dealer": 1, "Trustmark Dealer": 2}
    transmission_map = {"Manual": 0, "Automatic": 1}
    owner_map = {
        "First Owner": 0,
        "Second Owner": 1,
        "Third Owner": 2,
        "Fourth & Above Owner": 3,
        "Test Drive Car": 4,
    }

    car_data["fuel"] = car_data["fuel"].map(fuel_map)
    car_data["seller_type"] = car_data["seller_type"].map(seller_map)
    car_data["transmission"] = car_data["transmission"].map(transmission_map)
    car_data["owner"] = car_data["owner"].map(owner_map)

    car_data.reset_index(drop=True, inplace=True)

    feature_order = [
        "name", "year", "km_driven", "fuel", "seller_type",
        "transmission", "owner", "mileage", "engine", "max_power", "seats",
    ]

    X = car_data[feature_order]
    y = car_data["selling_price"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    r2 = metrics.r2_score(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    print(f"R2 score on test set: {r2:.4f}")
    print(f"Mean absolute error: {mae:,.0f}")

    with open("model/car_price_model.pkl", "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "brands": brands,
        "brand_to_code": brand_to_code,
        "fuel_map": fuel_map,
        "seller_map": seller_map,
        "transmission_map": transmission_map,
        "owner_map": owner_map,
        "feature_order": feature_order,
        "year_min": int(car_data["year"].min()),
        "year_max": int(car_data["year"].max()),
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "models_by_brand": models_by_brand,
        "model_defaults": model_defaults,
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model and metadata saved to model/")


if __name__ == "__main__":
    main()
