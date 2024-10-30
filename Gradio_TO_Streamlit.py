import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page title
st.set_page_config(page_title="Flower Predictor", layout="wide")

# Load and prepare the data
@st.cache_data  # This will cache the data loading
def load_data():
    iris_dataset = load_iris()
    X = iris_dataset['data'][:, [0, 1]]  # We use only sepal length and width
    y = iris_dataset['target']
    return X, y

# Load and train the model
@st.cache_resource  # This will cache the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return rf, X_train, accuracy

def predict_flower(model, sepal_length, sepal_width):
    """
    Predicts whether an Iris flower is a Versicolor, Virginica, or Setosa.
    """
    input_data = np.array([sepal_length, sepal_width]).reshape(1, -1)
    prediction = model.predict(input_data)
    
    classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    return classes[prediction[0]]

# Main app
def main():
    # Title and description
    st.title("Flower Predictor")
    st.markdown("""
    Enter your measurement of Sepal Length and Sepal Width to predict the class of flower.  
    Data source: Iris Flower Dataset; Model: Random Forest Classifier
    """)
    
    # Load data and train model
    X, y = load_data()
    model, X_train, accuracy = train_model(X, y)
    
    # Display model accuracy
    st.sidebar.header("Model Information")
    st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Input sliders
    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(X_train[:,0].min()),
            max_value=float(X_train[:,0].max()),
            value=float(X_train[:,0].mean()),
            step=0.1
        )
    
    with col2:
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(X_train[:,1].min()),
            max_value=float(X_train[:,1].max()),
            value=float(X_train[:,1].mean()),
            step=0.1
        )
    
    # Make prediction
    prediction = predict_flower(model, sepal_length, sepal_width)
    
    # Display prediction with styling
    st.markdown("### Prediction")
    st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>{prediction}</h2>", 
                unsafe_allow_html=True)
    
    # Add a visualization of the input point
    st.markdown("### Visualization")
    fig, ax = plt.subplots()
    
    # Plot training data points
    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], label=predict_flower(model, 0, 0))
    
    # Plot the current input point
    ax.scatter(sepal_length, sepal_width, color='red', marker='*', s=200, 
              label='Your Input')
    
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.legend()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()