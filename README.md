 
# Term Deposit Subscription Prediction

This project uses machine learning to predict whether a client will subscribe to a term deposit based on their profile and past marketing interactions. The model is trained on marketing campaign data from a banking institution.

## Project Overview

### Goal:
Help marketing teams identify potential clients more likely to subscribe to a term deposit, using predictive insights from operational data.

### Dataset:
- **Source:** [LimeWire](https://limewire.com/d/x7Hsa#Ml2ucdiZhL)
- **File Used:** `bank-additional-full.csv`
- **Size:** 41,188 rows × 21 columns

---

##  ML Pipeline

1. **Data Exploration & Cleaning**
   - Checked for nulls, outliers, class imbalance
   - Dropped leakage column `duration`
   - Capped extreme values in `campaign`
   
2. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Focused on 8 most impactful features for the app interface
   
3. **Model Training & Evaluation**
   - Models: Logistic Regression, Logistic + SMOTE, Random Forest
   - Evaluation metrics: Accuracy, Precision, Recall, F1 Score

---

##  Model Comparison

| Model                     | Accuracy | Recall (Subscribed) | Precision | F1 Score |
|--------------------------|----------|----------------------|-----------|----------|
| Logistic Regression      | 83%      | **65%**              | 37%       | **0.47** |
| Logistic + SMOTE         | 85%      | 42%                  | 35%       | 0.38     |
| **Random Forest**        | **90%**  | 29%                  | **57%**   | 0.38     |

>  Final model: **Random Forest**, deployed with top 8 features for a cleaner UI.

---

##  Streamlit App

Try the live demo:

👉 **[Launch App](https://your-username-your-app.streamlit.app)**

### User Inputs:
The app uses these 8 key inputs to predict subscription likelihood:
- Age
- Number of contacts during campaign
- Days since last contact
- Euribor 3-month rate
- Employment variation rate
- Number of employees
- Has housing loan?
- Has personal loan?

---

## 🗂️ Repository Structure

📁 Bank-_Term_Deposit/

│

├── app.py # Streamlit app

├── rf_model.pkl # Trained model (Random Forest)

├── model_columns.pkl # Feature list for model alignment

├── eda_modeling.ipynb # Jupyter notebook with EDA + model training

├── requirements.txt # Dependencies

└── README.md # Project summary


---

## ⚙️ How to Run Locally

```bash
# Create virtual env and activate it
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py

🙋 Author
Millicent Ama Agyir
Feel free to reach out via agyirnanagmail.com for questions or feedback!

---



