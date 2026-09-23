# Used Car Price Prediction

This repository contains a Machine Learning project that predicts the selling price of used cars based on various features such as manufacturing year, present price, kilometers driven, fuel type, seller type, transmission, and owner history[cite: 1].

## 📌 Project Overview

Predicting used car prices accurately helps buyers and sellers make informed financial decisions. Using data preprocessing, feature encoding, and a **Random Forest Regression** model, this project builds a pipeline to estimate prices and evaluates the prediction accuracy against actual data[cite: 1].

---

## 🎯 Objectives

- Perform data exploration, cleaning, and duplicate removal[cite: 1].
- Preprocess categorical features using One-Hot Encoding (`drop_first=True`)[cite: 1].
- Train a `RandomForestRegressor` model to predict car selling prices[cite: 1].
- Evaluate model performance using standard regression metrics: **MAE**, **MSE**, **RMSE**, and **R² Score**[cite: 1].
- Analyze prediction errors by comparing actual vs. predicted values[cite: 1].

---

## 📊 Dataset Details

The dataset (`car data.csv`) contains 301 records with the following attributes[cite: 1]:

| Feature | Description | Type |
| :--- | :--- | :--- |
| `Car_Name` | Name/brand of the car *(Dropped before training)* | Categorical |
| `Year` | Year of purchase | Numerical |
| `Present_Price` | Current ex-showroom price (in lakhs) | Numerical |
| `Kms_Driven` | Total kilometers driven | Numerical |
| `Fuel_Type` | Fuel type (`Petrol`, `Diesel`, `CNG`) | Categorical |
| `Seller_Type` | Type of seller (`Dealer`, `Individual`) | Categorical |
| `Transmission` | Gearbox type (`Manual`, `Automatic`) | Categorical |
| `Owner` | Number of previous owners | Numerical |
| **`Selling_Price`** | **Target Variable**: Price the car is being sold for | Numerical |

---

## 🛠 Tech Stack & Dependencies

- **Language:** Python 3.x
- **Libraries:**
  - `pandas` – Data manipulation & analysis
  - `numpy` – Numerical calculations
  - `scikit-learn` – Machine learning algorithms & model evaluation

To install the required dependencies:

```bash
pip install pandas numpy scikit-learn

```

---

## ⚙️ Workflow & Implementation

### 1. Data Cleaning

* Checked missing values: **0 missing values found**.


* Checked duplicates: **2 duplicate rows found** and dropped (reduced dataset size from 301 to 299 rows).



### 2. Feature Engineering & Preprocessing

* Removed `Car_Name` and target `Selling_Price` from feature matrix $X$.


* Applied One-Hot Encoding to categorical columns (`Fuel_Type`, `Seller_Type`, `Transmission`) using `drop_first=True`.



### 3. Model Training

* **Split:** 80% Training Data, 20% Testing Data (`random_state=42`).


* **Algorithm:** `RandomForestRegressor(n_estimators=100, random_state=42)`.



```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Fit model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

```

---

## 📈 Model Performance & Results

The model performance evaluated on the unseen test set yielded the following results:

| Evaluation Metric | Result |
| --- | --- |
| **Mean Absolute Error (MAE)** | `1.4664` |
| **Mean Squared Error (MSE)** | `12.2569` |
| **Root Mean Squared Error (RMSE)** | `3.5010` |
| **R² Score** | `0.5244` (52.44%) |

### Sample Predictions vs Actuals

```python
result["Error"] = result["Actual"] - result["Prediction"]

```

| Index | Actual Price | Predicted Price | Error |
| --- | --- | --- | --- |
| **283** | 8.99 | 9.4234 | -0.4334 |
| **267** | 8.35 | 8.2690 | +0.0810 |
| **166** | 0.45 | 0.4447 | +0.0053 |
| **9** | 7.45 | 6.9295 | +0.5205 |

---

## 💡 Key Takeaways & Future Enhancements

* The Random Forest model provides a reasonable baseline for estimating used car prices with an average error (MAE) of ~1.47 units.


* **Future Improvements:**
* Perform hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
* Feature engineering (e.g., calculating car age: `Current_Year - Year`).
* Experimenting with gradient boosting models like XGBoost, LightGBM, or CatBoost to improve $R^2$ score.



```


```
