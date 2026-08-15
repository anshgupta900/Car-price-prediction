import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

data = joblib.load("cars_price_model.pkl")
model = data["model"]
scaler = data["scaler"]
columns = data["columns"]
scaled = data["scaled"]

df = pd.read_csv("cars.csv")
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
df.dropna(subset=["Car Name", "Year"], inplace=True)
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])
df.reset_index(drop=True, inplace=True)
df["Year"] = df["Year"].astype(int)

st.title("🚗 Car Price Prediction")
st.write("Fill the form below and get an estimated price")
st.divider()

with st.form("car_form"):

    car_name = st.selectbox("Car Name", sorted(df["Car Name"].unique()))

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input(
            "Year",
            min_value=int(df["Year"].min()),
            max_value=int(df["Year"].max()),
            value=int(df["Year"].max())
        )
        owner = st.selectbox(
            "Owner",
            sorted(df["Owner"].dropna().unique()),
            format_func=lambda x: f"{x} previous owner(s)"
        )
        fuel = st.selectbox("Fuel Type", sorted(df["Fuel"].dropna().unique()))
        location = st.selectbox("Location", sorted(df["Location"].unique()))

    with col2:
        distance = st.number_input("Distance (km)", min_value=0, value=50000)
        drive = st.selectbox("Drive", sorted(df["Drive"].dropna().unique()))
        car_type = st.selectbox("Car Type", sorted(df["Type"].dropna().unique()))

    submitted = st.form_submit_button("🚗 Predict Price", use_container_width=True)

if submitted:
    input_data = pd.DataFrame({
        "Car Name": [car_name],
        "Year": [year],
        "Distance": [distance],
        "Owner": [owner],
        "Fuel": [fuel],
        "Location": [location],
        "Drive": [drive],
        "Type": [car_type]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["Car Name", "Owner", "Fuel", "Location", "Drive", "Type"],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(columns=columns, fill_value=0)

    if scaled:
        prediction = model.predict(scaler.transform(input_data))
    else:
        prediction = model.predict(input_data)

    st.success(f"💰 Estimated Car Price: ₹{prediction[0]:,.2f}")
