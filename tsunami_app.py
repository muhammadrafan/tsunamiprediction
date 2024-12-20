import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load supervised and unsupervised models
pickle_in_supervised = open('model_supervised_rf.pkl', 'rb')
supervised_model = pickle.load(pickle_in_supervised)

pickle_in_unsupervised = open('model_unsupervised_kmeans.pkl', 'rb')
unsupervised_model = pickle.load(pickle_in_unsupervised)

# Data loading and preprocessing
def load_and_preprocess_data():
    df = pd.read_csv('earthquake_data.csv')

    # Drop unnecessary columns
    df = df.drop(columns=['title', 'alert', 'country', 'continent', 'location', 'net'])

    # Convert date_time to year and month
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['Year'] = pd.DatetimeIndex(df["date_time"]).year
    df['Month'] = pd.DatetimeIndex(df["date_time"]).month
    df.drop('date_time', axis=1, inplace=True)

    # Label encode categorical variable
    lr = LabelEncoder()
    df['magType'] = lr.fit_transform(df['magType'])

    # Scale features
    sc = StandardScaler()
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    X_scaled = sc.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Handle imbalanced data
    sm = SMOTE(random_state=42)
    X_resample, y_resample = sm.fit_resample(X, y)
    df_balanced = pd.concat([pd.DataFrame(X_resample, columns=X.columns), pd.DataFrame(y_resample, columns=['tsunami'])], axis=1)

    return df_balanced, X_resample, y_resample

# Supervised prediction function
def prediction_supervised(input_data):
    pred = supervised_model.predict(input_data)
    proba = supervised_model.predict_proba(input_data)[:, 1]
    return pred, proba

# Unsupervised prediction function
def prediction_unsupervised(input_data):
    cluster = unsupervised_model.predict(input_data)
    return cluster

# Evaluation function for supervised model
def evaluate_model(X_test, y_test):
    y_pred = supervised_model.predict(X_test)
    y_pred_proba = supervised_model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    return metrics, y_pred, y_pred_proba

# Plot ROC curve
def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', label=f"ROC Curve (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tsunami', 'Tsunami'], yticklabels=['No Tsunami', 'Tsunami'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Streamlit application
def main():
    st.title("Tsunami Prediction from Earthquake Data")

    # HTML style for header
    html_temp = """
    <div style="background-color:darkblue; padding:13px; border-radius:15px; margin-bottom:20px;">
        <h1 style="color:white; text-align:center;">Tsunami Classifier ML App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Load and preprocess data
    df, X, y = load_and_preprocess_data()

    # Show dataframe
    if st.checkbox("Show Dataset"):
        st.dataframe(df.head())

    # Model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metrics, y_pred, y_pred_proba = evaluate_model(X_test, y_test)

    # Display metrics
    st.subheader("Model Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    col4.metric("F1 Score", f"{metrics['f1']:.2f}")

    # Plot options
    plot_option = st.selectbox("Choose a plot to display:", ["Select", "ROC AUC Curve", "Confusion Matrix"])
    if plot_option == "ROC AUC Curve":
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plot_roc_curve(fpr, tpr, metrics['roc_auc'])
    elif plot_option == "Confusion Matrix":
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)

    # Input prediksi
    st.subheader("Prediction Input")
    magnitude = st.number_input("Magnitude", value=float(df['magnitude'].mean()))
    cdi = st.number_input("CDI", value=float(df['cdi'].mean()))
    mmi = st.number_input("MMI", value=float(df['mmi'].mean()))
    sig = st.number_input("SIG", value=float(df['sig'].mean()))
    nst = st.number_input("NST", value=float(df['nst'].mean()))
    dmin = st.number_input("Dmin", value=float(df['dmin'].mean()))
    gap = st.number_input("Gap", value=float(df['gap'].mean()))
    magType = st.number_input("MagType", value=float(df['magType'].mean()))
    depth = st.number_input("Depth (km)", value=float(df['depth'].mean()))
    latitude = st.number_input("Latitude", value=float(df['latitude'].mean()))
    longitude = st.number_input("Longitude", value=float(df['longitude'].mean()))
    year = st.number_input("Year", value=int(df['Year'].mean()))
    month = st.number_input("Month", value=int(df['Month'].mean()))
    
    # Buat input menjadi DataFrame
    input_data_rf = pd.DataFrame([[magnitude, cdi, mmi, sig, nst, dmin, gap, magType, depth, latitude, longitude, year, month]],
                              columns=['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 
                                       'magType', 'depth', 'latitude', 'longitude', 'Year', 'Month'])
    input_data_kmeans = pd.DataFrame([[magnitude, depth, latitude, longitude]],
                              columns=['magnitude', 'depth', 'latitude', 'longitude'])

    # Prediksi
    if st.button("Predict (Supervised Model)"):
        result, proba = prediction_supervised(input_data_rf)
        st.success(f'Prediction (Supervised): {"Tsunami" if result[0] == 1 else "No Tsunami"}')
        st.info(f'Probability: {proba[0]:.2f}')

    if st.button("Predict (Unsupervised Model)"):
        cluster = prediction_unsupervised(input_data_kmeans)
        st.success(f'Cluster: {cluster[0]}')

if __name__ == '__main__':
    main()
