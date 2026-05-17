
import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

def cap_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[column].clip(lower_bound, upper_bound)

df['sepal width (cm)'] = cap_outliers('sepal width (cm)')

x = df.drop(columns=["target"])
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

knn_classifier = KNeighborsClassifier(
    n_neighbors=3,
    metric='hamming'
)

knn_classifier.fit(x_train, y_train)

y_pred = knn_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

st.title("Iris Flower Prediction Using KNN")

st.success(f"Model Accuracy: {accuracy:.2f}")

sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):

    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    prediction = knn_classifier.predict(input_data)[0]

    if prediction == 0:
        st.success("Predicted Flower: Setosa")

    elif prediction == 1:
        st.success("Predicted Flower: Versicolor")

    else:
        st.success("Predicted Flower: Virginica")
