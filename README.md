# 🚗 Car Price Prediction Web Application

A machine learning-powered web application that predicts the resale price of used cars based on vehicle specifications and ownership details. The application uses a trained **Linear Regression** model and provides real-time predictions through a modern **FastAPI** backend and an interactive web interface.

---

## 📖 Overview

This project predicts the estimated selling price of used cars by applying machine learning techniques to historical vehicle data. The model is trained using data preprocessing, feature engineering, and regression analysis. The trained model is then integrated into a FastAPI application for real-time price prediction.

---

## ✨ Features

* Predicts used car resale prices instantly.
* FastAPI backend with REST API.
* Responsive and user-friendly web interface.
* Automatic brand and model selection.
* Model-specific default specifications.
* Real-time prediction with formatted price output.
* Input validation using Pydantic.
* Pre-trained machine learning model for efficient inference.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Linear Regression

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* HTML5
* CSS3
* JavaScript

### Model Serialization

* Pickle
* JSON

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📊 Dataset

The project uses the **Cardetails.csv** dataset containing information such as:

* Car Brand
* Manufacturing Year
* Selling Price
* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission
* Owner Type
* Mileage
* Engine Capacity
* Maximum Power
* Number of Seats

---

## 🧹 Data Preprocessing

The dataset was cleaned and prepared before training the machine learning model using the following steps:

* Removed the **torque** feature since it was not used for prediction.
* Removed rows containing missing values.
* Removed duplicate records to improve data quality.
* Converted text-based numerical values such as:

  * `23.4 kmpl → 23.4`
  * `1248 CC → 1248`
  * `74 bhp → 74`
* Extracted the vehicle brand from the complete car name.
* Encoded categorical variables into numerical values:

  * Brand
  * Fuel Type
  * Seller Type
  * Transmission
  * Owner Type
* Reset the dataset index after preprocessing.

---

## ⚙️ Machine Learning Workflow

1. Load dataset
2. Data cleaning and preprocessing
3. Feature engineering
4. Encode categorical variables
5. Train-test split (80% training, 20% testing)
6. Train Linear Regression model
7. Evaluate model performance
8. Save trained model using Pickle
9. Generate metadata for frontend integration
10. Deploy model using FastAPI

---

## 📈 Model Evaluation

The model performance is evaluated using:

* R² Score
* Mean Absolute Error (MAE)

These evaluation metrics are stored inside the generated metadata file and displayed through the application.

---

## 📂 Project Structure

```text
Car-price-prediction-project/
│
├── model/
│   ├── car_price_model.pkl
│   └── metadata.json
│
├── templates/
│   └── index.html
│
├── static/
│
├── app.py
├── train_model.py
├── Cardetails.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Kolusuissak05/Car-price-prediction-project.git
```

### Navigate to the project

```bash
cd Car-price-prediction-project
```

### Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the FastAPI application

```bash
uvicorn app:app --reload
```

Open your browser and visit:

```
http://127.0.0.1:8000
```

---

## 🔮 Prediction Process

1. User enters vehicle details.
2. FastAPI validates the input.
3. Categorical values are converted into numerical codes using stored metadata.
4. Features are arranged in the same order used during training.
5. The trained Linear Regression model predicts the resale price.
6. The predicted price is returned and displayed in the web interface.

---

## 📌 Future Improvements

* Compare multiple machine learning algorithms.
* Hyperparameter tuning.
* Cloud deployment.
* User authentication.
* Price trend visualization.
* Vehicle image upload.
* Advanced feature engineering.
* Model retraining pipeline.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 👨‍💻 Author

**Issak Kolusu**

GitHub: https://github.com/Kolusuissak05

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.
