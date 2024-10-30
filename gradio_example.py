import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gradio as gr

# Load the data and split into data and labels
from sklearn.datasets import load_iris
iris_dataset = load_iris()
# We use only The 0'th and 1'st columns.
# These correspond to sepal length and width.
X = iris_dataset['data'][:, [0, 1]]
y = iris_dataset['target']

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Import and train a machine learning model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

def predict_flower(sepal_length, sepal_width):
    """
    Predicts whether an Iris flower is a Versicolor, Virginica, or Setosa.

    Args:
        sepal_length (float): The flower's average sepal length in cm.
        sepal_width (float): The flower's average sepal width in cm.

    Returns:
        str: "Virginica", "Setosa", or "Versicolor".
    """

    # Reshape the input data for prediction
    input_data = np.array([sepal_length, sepal_width]).reshape(1, -1)

    # Get the prediction
    prediction = rf.predict(input_data)

    # Return the result
    if prediction[0] == 0:
        return "Setosa"
    elif prediction[0] == 1:
        return "Versicolor"
    elif prediction[0] == 2:
        return "Virginica"
    else:
        raise Exception


# Set the minimum, maximum, and default values for the sliders
# This is optional
sl_min = X_train[:,0].min().round(2)
sl_max = X_train[:,0].max().round(2)
sl_default = X_train[:,0].mean().round(2)

sw_min = X_train[:,1].min().round(2)
sw_max = X_train[:,1].max().round(2)
sw_default = X_train[:,1].mean().round(2)

# Create the interface
iface = gr.Interface(
    fn=predict_flower,
    inputs=[
        gr.components.Slider(minimum=sl_min, maximum=sl_max,
                             value=sl_default, label="Sepal length"),
        gr.components.Slider(minimum=sw_min, maximum=sw_max,
                             value=sw_default, label="Sepal width"),
    ],
    outputs=gr.components.Textbox(label="Prediction"),
    title="Flower Predictor",
    description="""Enter your measurement of Sepal Length and Sepal Width to
    predict the class of flower.
    Data source: Iris Flower Dataset; Model: Random Forest Classifier""",
)

# Launch the interface
iface.launch(share=True,debug=True)