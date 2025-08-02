import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title("Customer Churn Prediction Dashboard")

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Preprocess
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show metrics
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots()
importances.sort_values().plot(kind='barh', ax=ax2)
st.pyplot(fig2)
