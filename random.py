import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import plotly.express as px
from PIL import Image
import hiplot as hip
import random
from scipy import linalg

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

image = st.image("wine.png", use_column_width=True)

st.write(
    "For ages, wine has been a big deal in our lives and cultures. Long ago, people even thought it helped them talk to their Gods because it made them feel different. That's why they valued it so much in their temples and ceremonies. Nowadays, most of us just see wine as something nice to have now and then. Its story is pretty fascinating and has seen a lot of ups and downs. But you know what? Wine's journey isn't over yet. It's still going strong and will keep changing as time goes on. We'll find new ways to enjoy it, and each way will have its own special meaning for us.The taste of red wine depends on lots of things. Acidity, from fixed, volatile, and citric acids, defines its freshness and food compatibility. Residual sugar levels determine sweetness, while chlorides subtly impact taste. Sulfur dioxide preserves and influences aroma. Density shapes body and texture, pH affects stability, and sulphates safeguard against spoilage. Finally, alcohol content contributes to overall perception. All these things together make each red wine special and tasty in its own way.")

df = pd.read_csv("winequality-red.csv")
data_tab, story_tab, pca_tab, model_tab, class_tab = st.tabs(
    ["Dataset", "Story", "PCA", "Model Prediction", "Classification"])
with data_tab:
    st.write("## About Dataset")
    st.write(
        "Following datasets consists of 11 different features such as fixed acid, volatile acid, total sulfur, pH and so on with 1599 observations in total. Based on these value the quality of red wine is determined and a numeric number is assigned from 0 worst to 10 best. ")
    b1 = st.button(":red[Show dataset]")
    if b1:
        st.dataframe(df)
    if st.button(":red[Hide dataset]"):
        b1 = False

with pca_tab:
    st.title("PCA Analysis on multiple data")
    column_names = [''] + df.columns.tolist()
    st.write("Please select different category for each box")
    x = st.selectbox("Select Category for PCA", column_names, key = "option1" )
    y = st.selectbox("Select Category for PCA", column_names, key = 'option2')
    z = st.selectbox("Select Category for PCA", column_names, key = "option3")

    if len(set([x, y, z])) != len([x, y, z]):
        st.write("## Please select three different columns, at least two of the column selected are identical.")
    else:

        selected_columns = df[[x, y, z]]
        centered_data = selected_columns - selected_columns.mean()
        U, Sigma, VT = np.linalg.svd(centered_data)
        m = len(U)
        n = len(VT)
        sigma = np.zeros([m, n])
        sigma1 = np.copy(sigma)
        for i in range(1):
            sigma1[i, i] = Sigma[i]
        X_1D = U @ sigma1 @ VT
        sigma2 = np.copy(sigma)
        for i in range(2):
            sigma2[i, i] = Sigma[i]
        X_2D = U @ sigma2 @ VT

        fig_3D = px.scatter_3d(centered_data, x=x, y=y, z=z,
                               title='3D Scatter Plot of Data')
        st.plotly_chart(fig_3D)
        fig_2D = px.scatter_3d(pd.DataFrame(X_2D, columns=[x, y, z]),
                               x=x, y=y, z=z,
                               title='2D Scatter Plot of Data')
        st.plotly_chart(fig_2D)
        fig_1D = px.scatter_3d(pd.DataFrame(X_1D, columns=[x, y, z]),
                               x=x, y=y, z=z,
                               title='1D Scatter Plot of Data')

        st.plotly_chart(fig_1D)

with model_tab:
    st.write("## Predict the quality of red wine")

    X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
    y_quality = df['quality']
    X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(X, y_quality, test_size=0.1,
                                                                            random_state=3)
    model_quality = LinearRegression()
    model_quality.fit(X_train_qual, y_train_qual)
    predictions_quality = model_quality.predict(X_test_qual)
    mse_quality = mean_squared_error(y_test_qual, predictions_quality)
    st.write(f"Mean Squared Error for Sleep Quality:", mse_quality)
    st.title("Predict Sleep Parameters")
    st.title('Wine Quality Data Input')

    col1, col2, col3 = st.columns(3)

    with col1:
        i1 = st.slider("Fixed Acidity", min_value=0.0, max_value=15.9, format="%.2f")
        i4 = st.slider("Residual Sugar", min_value=0.0, max_value=16.0, format="%.2f")
        i7 = st.slider("Total Sulfur Dioxide", min_value=0, max_value=300)
        i10 = st.slider("Sulphates", min_value=0.0, max_value=2.0, format="%.2f")

    with col2:
        i2 = st.slider("Volatile Acidity", min_value=0.0, max_value=1.58, format="%.2f")
        i5 = st.slider("Chlorides", min_value=0.0, max_value=0.61, format="%.2f")
        i8 = st.slider("Density", min_value=0.9, max_value=1.0, format="%.4f")
        i11 = st.slider("Alcohol", min_value=0.0, max_value=20.0, format="%.2f")

    with col3:
        i3 = st.slider("Citric Acid", min_value=0.0, max_value=1.0, format="%.2f")
        i6 = st.slider("Free Sulfur Dioxide", min_value=0, max_value=72)
        i9 = st.slider("pH", min_value=0.0, max_value=4.5, format="%.2f")

    input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

    if st.button('Predict'):
        predicted_quality = model_quality.predict(input_data)[0]
        if predicted_quality >= 7:
            cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2, 2, 2, 2])
            cc2.write("The quality is good")
            cc4.image("good.png", use_column_width=True)
        else:
            cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2, 2, 2, 2])
            cc2.write("The quality is bad")
            cc4.image("bad.png", use_column_width=True)

with class_tab:
    st.write("## Try different models for testing the quality of red wine")

    # ... (Your existing code)

    col_labels = [
        ["Fixed Acidity", "Volatile Acidity", "Citric Acid"],
        ["Residual Sugar", "Chlorides", "Free Sulfur Dioxide"],
        ["Total Sulfur Dioxide", "Density", "pH"],
        ["Sulphates", "Alcohol", ""]
    ]

    col_values = [
        [0.0, 0.0, 0.0],
        [15.9, 1.58, 1.0],
        [15.5, 0.61, 72],
        [289, 1.0, 14.0],
        [2.0, 20.0, None]
    ]

    cc_cols = [st.columns(5) for _ in range(5)]

    for i in range(5):
        with cc_cols[i][1]:
            for j in range(3):
                if col_values[i][j] is not None:
                    value = col_values[i][j]
                    if isinstance(value, float):
                        value_fmt = "%.2f"
                    else:
                        value_fmt = "%d" if isinstance(value, int) else ""
                    slider_value = st.slider(col_labels[i][j], min_value=0.0, max_value=value, format=value_fmt,
                                             key=f"i{i * 3 + j + 1}")

    input_data = np.array([slider_value for cc in cc_cols for slider_value in cc[1].slider_value]).reshape(1, -1)

    if st.button('Predict', key='predict_button'):
        predicted_quality = model_quality.predict(input_data)[0]
        cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2, 2, 2, 2])
        if predicted_quality >= 10:
            cc2.write("The quality is good")
            cc4.image("good.png", use_column_width=True)
        else:
            cc2.write("The quality is bad")
            cc4.image("bad.png", use_column_width=True)

    st.write("## Try different models for testing the quality of red wine")

    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)

    with col1:
        i1 = st.slider("Fixed Acidity", min_value=0.0, max_value=15.9, format="%.2f", key="i1")
        i4 = st.slider("Residual Sugar", min_value=0.0, max_value=15.5, format="%.2f", key="i4")
        i7 = st.slider("Total Sulfur Dioxide", min_value=0.0, max_value=289.0, key="i7")
        i10 = st.slider("Sulphates", min_value=0.0, max_value=2.0, format="%.2f", key="i10")

    with col2:
        i2 = st.slider("Volatile Acidity", min_value=0.0, max_value=1.58, format="%.2f", key="i2")
        i5 = st.slider("Chlorides", min_value=0.0, max_value=0.61, format="%.2f", key="i5")
        i8 = st.slider("Density", min_value=0.0, max_value=1.0, format="%.4f", key="i8")
        i11 = st.slider("Alcohol", min_value=0.0, max_value=20.0, format="%.2f", key="i11")

    with col3:
        i3 = st.slider("Citric Acid", min_value=0.0, max_value=1.0, format="%.2f", key="i3")
        i6 = st.slider("Free Sulfur Dioxide", min_value=0, max_value=72, key="i6")
        i9 = st.slider("pH", min_value=0.0, max_value=14.0, format="%.2f", key="i9")

    input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

    if st.button('Predict', key='predict_button'):
        predicted_quality = model_quality.predict(input_data)[0]
        if predicted_quality >= 10:
            cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2, 2, 2, 2])
            cc2.write("The quality is good")
            cc4.image("good.png", use_column_width=True)
        else:
            cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2, 2, 2, 2])
            cc2.write("The quality is bad")
            cc4.image("bad.png", use_column_width=True)