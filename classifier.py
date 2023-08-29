import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load and preprocess the data
df = pd.read_csv('XSS_dataset.csv', encoding='utf-8-sig')
X = df['Sentence']
y = df['Label']

vectorizer = CountVectorizer(min_df=2, max_df=0.8)
X = vectorizer.fit_transform(X.values.astype('U')).toarray()


# Load the trained models
lr_model = joblib.load('logistic_regression_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

# Function to classify the script
def classify_input():
    script = script_entry.get()

    # Preprocess the input script
    feature_vector = vectorizer.transform([script]).toarray()

    # Predict using logistic regression model
    lr_pred = lr_model.predict(feature_vector)
    if lr_pred[0] == 1:
        lr_result_label.config(text="Logistic Regression: The script is malicious")
    else:
        lr_result_label.config(text="Logistic Regression: The script is not malicious")

    # Predict using Decision Tree model
    dt_pred = dt_model.predict(feature_vector)
    if dt_pred[0] == 1:
        dt_result_label.config(text="Decision Tree: The script is malicious")
    else:
        dt_result_label.config(text="Decision Tree: The script is not malicious")

# Create the GUI
window = tk.Tk()
window.title("Malicious Script Classification")
window.geometry("400x300")

script_label = tk.Label(window, text="Enter a script:")
script_label.pack()

script_entry = tk.Entry(window, width=40)
script_entry.pack()

classify_button = tk.Button(window, text="Classify", command=classify_input)
classify_button.pack()

lr_result_label = tk.Label(window, text="")
lr_result_label.pack()

dt_result_label = tk.Label(window, text="")
dt_result_label.pack()

window.mainloop()
