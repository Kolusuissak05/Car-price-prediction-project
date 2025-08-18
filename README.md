# Car-price-prediction-project

# 📌 Overview

This project predicts the selling price of cars based on features such as brand, year, mileage, fuel type, transmission, and other specifications.  
It uses machine learning techniques to help buyers and sellers estimate fair car prices.

# 📂 Dataset

**Source:** Cardetails.csv (contains 8128 rows, 12 columns)  
**After preprocessing and cleaning:** 6718 rows, 12 columns

**Features include:**
- name (Car brand)
- year (Year of manufacture)
- km_driven (Kilometers driven)
- fuel (Fuel type)
- seller_type (Dealer/Individual/Trustmark Dealer)
- transmission (Manual/Automatic)
- owner (Ownership details)
- mileage, engine, max_power, seats
- selling_price (Target variable)

# 🛠️ Libraries Used

- Pandas, NumPy – Data manipulation  
- Matplotlib, Seaborn – Visualization  
- Scikit-learn – Preprocessing, Linear Regression, Model Evaluation

# 🔍 Data Preprocessing

- Dropped irrelevant columns (torque)
- Cleaned numeric columns (mileage, engine, max_power)
- Encoded categorical columns (fuel, transmission, seller_type, owner, name)
- Handled missing values and duplicates

**Final dataset:** 6718 rows, 12 features

# 📊 Exploratory Data Analysis (EDA)

- Visualized selling price distributions by fuel type, seller type, transmission
- Checked correlations using heatmap
- Observed that year, engine, and max_power are strongly correlated with selling price

# 🤖 Model Building

- Split data into train (80%) and test (20%) sets
- Applied Linear Regression model
- Trained and tested model on cleaned dataset

**Example prediction:**
```python
input_data = [[6,2016,11000,0,0,0,3,17.8,1248.0,75.0,5.0]]
model.predict(input_data) 
# Output: ~464,258
```

# 📈 Results

- Model successfully predicts approximate car prices
- Plotted Actual vs Predicted Prices with regression line
- Future scope: Try advanced models (Random Forest, XGBoost) for higher accuracy

# 🚀 Future Improvements

- Deploy the model as a web app using Flask or Streamlit
- Add a user interface for inputting car details and predicting prices
- Experiment with hyperparameter tuning and ensemble models
- Deploy on cloud (Heroku, AWS, or Render)
