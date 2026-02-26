import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Home", "EDA", "Model Training", "Prediction", "Model Comparison"])

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# ================= HOME =================
if page == "Home":
    st.title("Customer Segmentation using RFM")
    st.write("Black Friday Sales Dataset")
    st.write("Total Records:", df.shape[0])
    st.write("Total Features:", df.shape[1])
    st.write(df.head())

# ================= EDA =================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Gender Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Gender', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Purchase Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Purchase'], bins=30)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# ================= MODEL TRAINING =================
elif page == "Model Training":
    st.title("Clustering + Classification")

    # RFM
    rfm = df.groupby('User_ID').agg({
        'Purchase': ['count','sum']
    })

    rfm.columns = ['Frequency','Monetary']
    rfm.reset_index(inplace=True)
    rfm['Recency'] = rfm['Frequency'].max() - rfm['Frequency']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

    sil_score = silhouette_score(rfm_scaled, rfm['Segment'])
    st.write("Silhouette Score:", sil_score)

    # Classification
    X = rfm[['Recency','Frequency','Monetary']]
    y = rfm['Segment']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    knn.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    rf.fit(X_train,y_train)

    st.success("Models Trained Successfully")

# ================= PREDICTION =================
elif page == "Prediction":
    st.title("Predict Customer Segment")

    recency = st.number_input("Recency", min_value=0)
    frequency = st.number_input("Frequency", min_value=0)
    monetary = st.number_input("Monetary", min_value=0)

    if st.button("Predict"):
        # Temporary simple prediction logic
        if monetary > 10000:
            segment = "Premium Buyer"
        elif frequency > 5:
            segment = "Regular Customer"
        else:
            segment = "Occasional Visitor"

        st.success(f"Predicted Segment: {segment}")

# ================= MODEL COMPARISON =================
elif page == "Model Comparison":
    st.title("Model Comparison")

    st.write("Model Performance Comparison (Example Values)")

    comparison = pd.DataFrame({
        "Model": ["KNN", "Decision Tree", "Random Forest"],
        "Accuracy": [0.85, 0.88, 0.92],
        "Macro F1 Score": [0.84, 0.87, 0.91]
    })

    st.table(comparison)