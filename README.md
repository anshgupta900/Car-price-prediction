# 🚗 Car Price Prediction

A web app that predicts the resale price of a used car based on its details — instantly, right in your browser.

Enter a car's name, year, mileage, fuel type, and a few other details, and get an estimated price in ₹ — powered by a machine learning model trained on real car sale data.

---

## ✨ Features

- 🎯 Instant price prediction from simple form inputs
- 📋 Dropdowns populated from real car listings — no manual typing
- 🧠 Powered by a Random Forest regression model
- ⚡ Clean, minimal interface built with Streamlit

---

## 🖥️ Try It Out

Fill in:
- Car Name
- Year
- Distance driven (km)
- Number of previous owners
- Fuel type
- Location
- Drive type
- Car type

Click **Predict Price** and get an instant estimate.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Model:** scikit-learn (Random Forest Regressor)
- **Data handling:** pandas
- **Language:** Python

---

## 🚀 Getting Started

Clone the repo, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local link Streamlit shows you, and you're in.

---

## 📁 Project Structure

```
├── app.py                   # The Streamlit app
├── cars.csv                 # Car listings dataset
├── cars_price_model.pkl     # Pre-trained prediction model
├── Train_model.ipynb        # Notebook used to train the model
└── requirements.txt         # Python dependencies
```

---

## 📊 About the Model

The model was trained on thousands of real used car listings, learning how price relates to factors like age, mileage, fuel type, and location. It uses a Random Forest — an ensemble of decision trees — to make its predictions.

---

## 📝 License

This project is open for learning and personal use.
