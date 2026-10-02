# Student Placement Prediction

A Streamlit app for predicting student placement using machine learning models.

## Features

- Upload a placement dataset in CSV format.
- Validate dataset schema and missing values.
- Train models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost.
- Enter student profile fields and get a placement prediction.
- View accuracy, precision, recall, and F1 score.
- Show placement distribution, correlation heatmap, and feature importance.
- Download prediction reports.

## Run the app

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Streamlit:

```bash
streamlit run app.py
```

## Supported datasets

- `placementdata.csv` sample dataset
- `Placement_Data_Full_Class.csv` sample dataset

The app detects the uploaded dataset schema and prepares data for model training.
