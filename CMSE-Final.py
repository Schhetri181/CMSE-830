import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import plotly.express as px
from PIL import Image
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

st.set_page_config( page_title='Wine Quality Analysis - Machine Learning Approach',page_icon=':house:',layout='wide')
st.title("Wine Quality Analysis - Machine Learning Approach")
left_column, center_colum, right_column = st.columns([4, 1, 4])

with left_column:
    image1 = st.image("vineyard_wine.webp", use_column_width=True)
    st.markdown(
        "<p style='text-align: justify;'>""<strong>" "People love good wine for many reasons. First, it is about how it tastes "
        "and makes us feel. A really good wine has amazing flavors and smells that dance on our tongues and noses. "
        "It's like a work of art, showing off the grape's unique taste, the land it grew in, and how carefully it was"
        " made by the winemaker. But wine isn't just about how it tastes. It brings people together. Sharing a great "
        "bottle of wine often turns into a special moment where friends chat, laugh, and create memories. Plus, wine "
        "is full of history and tradition. It's been a part of ceremonies and important events for centuries, carrying"
        " stories and meanings that connect us to our past. So, enjoying a good wine isn't just about drinking; it's "
        "like stepping into a world that's all about pleasure, togetherness, and our rich history.""<strong>"
        "</p>",
        unsafe_allow_html=True
    )

with right_column:
    st.markdown(
        "<p style='text-align: justify;'>""<strong>"
        "Machine learning has completely transformed the way we approach wine quality. It's like having a super-smart"
        " detective that looks at a ton of information about wine, like how sour or sweet it is, what chemicals are in"
        " it, and where the grapes were grown. With all this data, machine learning learns patterns and connections, "
        "kind of like how it learns from examples. This helps it predict how great a wine might turn out to be even "
        "before it's bottled. This prediction power is a game-changer for winemakers because they can use this insight"
        " to tweak things during the winemaking process, making the best possible wine. Also, machine learning helps "
        "in quality checks. It looks at the taste, smell, and other details of wine to make sure it matches what's "
        "expected. This webapp will help you to analyze the quality of a wine based on its features both qualitatively"
        " and quantitatively using different models. If there's something off, it can catch it early so that the "
        "wine turns out just perfect. In short, "
        "This webapp is like a wizard helping winemakers understand wine in a super-smart way, making sure every "
        "bottle is top-notch.""<strong>"
        "</p>",
        unsafe_allow_html=True
    )
    image2 = st.image("wine.png", use_column_width=True)

df = pd.read_csv("winequality-red.csv")
data_tab, story_tab, pca_tab, model_tab, class_tab, Bio_tab = st.tabs(
    ["Dataset", "Story", "PCA", "Wine quality - Prediction", "Wine Quality -Classification", "Bio"])
with data_tab:
    st.write("## About Dataset")
    st.write(
        "<p style='text-align: justify;'>"
        "Following datasets consists of 11 different features such as fixed acid, volatile acid, total sulfur, pH and"
        " so on with 1599 observations in total. Based on these value the quality of red wine is determined and a "
        "numeric number is assigned from 0 worst to 10 best."
        "</p>",

        unsafe_allow_html=True
    )

    b1 = st.button(":red[Show dataset]")
    if b1:
        st.dataframe(df)
    if st.button(":red[Hide dataset]"):
        b1 = False

    link = "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009"
    display_text = "Kaggle"

    st.write(f"##### Source - [{display_text}]({link})")

with story_tab:
    right_column, left_column = st.columns([1,1])
    with right_column:

        st.markdown(
            "<p style='text-align: justify;'>"
            " For ages, wine has been a big deal in our lives and cultures. Long ago, people even thought it helped"
            " them talk to their Gods because it made them feel different. That's why they valued it so much in their "
            "temples and ceremonies. Nowadays, most of us just see wine as something nice to have now and then. Its story"
            " is pretty fascinating and has seen a lot of ups and downs. But you know what? Wine's journey isn't over yet."
            " It's still going strong and will keep changing as time goes on..</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: justify;'>"
            "  During the Middle Ages, wine was a "
            "symbol of social status and wealth. Monasteries played a crucial role in preserving and advancing winemaking"
            " knowledge. Monks meticulously tended vineyards and refined winemaking methods, safeguarding this knowledge "
            "through written records. Wine became an integral part of daily life, from the noble courts to the humblest "
            "taverns..</p>",
            unsafe_allow_html=True
        )



        image1 = st.image("ancient_wine.webp", width=400)
    with left_column:
        st.markdown(
            "<p style='text-align: justify;'>"
            "In modern times, winemaking has evolved into a precise science, merging tradition with innovation."
            " Advances in technology, such as temperature-controlled fermentation and precise harvesting equipment, have "
            "revolutionized the industry. Winemakers employ modern analytics to understand and manipulate factors like "
            "acidity, sugar levels, and tannins, creating wines that cater to evolving consumer preferences</p>",
            unsafe_allow_html=True
        )

        image2 = st.image("wine_social.jpg",use_column_width=True)

        st.markdown(
            "<p style='text-align: justify;'>"
            " The taste "
            "of wine depends on lots of things. Acidity makes it feel fresh and good with certain foods. Sugar levels "
            "decide how sweet or dry it is. Some small elements, like chlorides, affect how it tastes subtly. There's "
            "something called sulfur dioxide that keeps its smell nice. How thick or smooth it feels in your mouth is "
            "because of its density. The acidity also helps it stay good for a long time. Depending upon the alcohol "
            "content, its strongness in taste also varies. All these things together make each wine special and tasty"
            " in its own way.</p>",
            unsafe_allow_html=True
        )



with pca_tab:
    st.title("Principal Component Analysis on multiple data")
    column_names = df.columns.tolist()
    st.write("Please select different category for each box")
    col1, col2, col3 = st.columns(3)

    # Select boxes in a single row
    with col1:
        x = st.selectbox("Select Category for PCA (X-axis)", column_names, key="option1")

    with col2:
        y = st.selectbox("Select Category for PCA (Y-axis)", column_names, key="option2")

    with col3:
        z = st.selectbox("Select Category for PCA (Z-axis)", column_names, key="option3")

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

        col1, col2, col3 = st.columns(3)
        with col1:
            fig_3D = px.scatter_3d(centered_data, x=x, y=y, z=z,
                                   title='3D Scatter Plot of Data')
            st.plotly_chart(fig_3D)

        with col2:
            fig_2D = px.scatter_3d(pd.DataFrame(X_2D, columns=[x, y, z]),
                                   x=x, y=y, z=z,
                                   title='2D Scatter Plot of Data')
            st.plotly_chart(fig_2D)

        with col3:
            fig_1D = px.scatter_3d(pd.DataFrame(X_1D, columns=[x, y, z]),
                                   x=x, y=y, z=z,
                                   title='1D Scatter Plot of Data')
            st.plotly_chart(fig_1D)

with model_tab:
    st.write("# Predict the quality of red wine")

    X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
    y_quality = df['quality']

    test_size1 = st.number_input("Test Size (as a decimal)", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                                 key = "size1")

    X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(X, y_quality, test_size=test_size1,
                                                                            random_state=3)
    model_quality = LinearRegression()
    model_quality.fit(X_train_qual, y_train_qual)
    predictions_quality = model_quality.predict(X_test_qual)
    mse_quality = mean_squared_error(y_test_qual, predictions_quality)
    st.write("### Wine Quality Data Input")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        i1 = st.slider("Fixed Acidity", min_value=0.0, max_value=15.9, format="%.2f")
        i4 = st.slider("Residual Sugar", min_value=0.0, max_value=16.0, format="%.2f")
        i9 = st.slider("pH", min_value=0.0, max_value=4.5, format="%.2f")

    with col2:
        i7 = st.slider("Total Sulfur Dioxide", min_value=0, max_value=300)
        i10 = st.slider("Sulphates", min_value=0.0, max_value=2.0, format="%.2f")

    with col3:
        i2 = st.slider("Volatile Acidity", min_value=0.0, max_value=1.58, format="%.2f")
        i5 = st.slider("Chlorides", min_value=0.0, max_value=0.61, format="%.2f")

    with col4:
        i8 = st.slider("Density", min_value=0.9, max_value=1.0, format="%.4f")
        i11 = st.slider("Alcohol", min_value=0.0, max_value=20.0, format="%.2f")

    with col5:
        i3 = st.slider("Citric Acid", min_value=0.0, max_value=1.0, format="%.2f")
        i6 = st.slider("Free Sulfur Dioxide", min_value=0, max_value=72)

    input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

    if st.button('Predict'):

        predicted_quality = model_quality.predict(input_data)[0]
        predicted_quality = int(predicted_quality)
        st.write(f"Wine quality is :", predicted_quality, font_size=24)
        if predicted_quality >= 6:
            st.write("The quality is good")
            st.image("good.png", width = 400)
        else:
            st.write("The quality is bad")
            st.image("bad.png", width = 400)

with class_tab:

    st.write("## Try different models for testing the quality of red wine")
    X = df.drop('quality', axis=1)
    y = df['quality']

    col1, col2, col3 = st.columns(3)

    with col1:
        i1 = st.slider("Fixed Acidity", min_value=0.0, max_value=15.9, format="%.2f", key="i1")
        i4 = st.slider("Residual Sugar", min_value=0.0, max_value=15.5, format="%.2f", key="i4")
        i7 = st.slider("Total Sulfur Dioxide", min_value=0.0, max_value=289.0, key="i7")
        i10 = st.slider("Sulphates", min_value=0.0, max_value=2.0, format="%.2f", key="i10")

    with col2:
        i2 = st.slider("Volatile Acidity", min_value=0.0, max_value=1.58, format="%.2f", key="i2")
        i5 = st.slider("Chlorides", min_value=0.0, max_value=0.61, format="%.2f", key="i5")
        i8 = st.slider("Density", min_value=0.9, max_value=1.0, format="%.4f", key="i8")
        i11 = st.slider("Alcohol", min_value=0.0, max_value=20.0, format="%.2f", key="i11")

    with col3:
        i3 = st.slider("Citric Acid", min_value=0.0, max_value=1.0, format="%.2f", key="i3")
        i6 = st.slider("Free Sulfur Dioxide", min_value=0, max_value=72, key="i6")
        i9 = st.slider("pH", min_value=0.0, max_value=14.0, format="%.2f", key="i9")
    test_size = st.number_input("Test Size (as a decimal)", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                                key="size22")
    x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(X, y, test_size=test_size,
                                                                        random_state=0)
    model_type = st.selectbox(
        "Model",
        (
            "K-Neighbors Classifier",
            "Logistic Regression",
            "Random Forest Classifier",
            "Decision Tree Classifier",
        )
    )
    if model_type == "K-Neighbors Classifier":
        modell = KNeighborsClassifier()
        neighbors_num = st.slider("Number of neighbors", value=3, min_value=1, max_value=30, step=1)
        modell = KNeighborsClassifier(n_neighbors=neighbors_num)
        modell.fit(x_cat_train, y_cat_train)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.title("Confusion Matrix")
            st.pyplot(fig)


        c1, c2 = st.columns(2)
        with c2:

            plot_confusion_matrix()
        with c1:



            input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

            if st.button('Predict', key='predict_button1'):
                predicted_quality = model_quality.predict(input_data)[0]
                if predicted_quality >= 10:
                    st.write("The quality is good")
                    st.image("good.png", width=400)
                else:
                    st.write("The quality is bad")
                    st.image("bad.png", width=400)

                accuracy = accuracy_score(y_cat_test, y_cat_pred)
                st.markdown(f"### Prediction Accuracy: {accuracy * 100:.2f}%")


    if model_type == "Logistic Regression":
        modell = LogisticRegression()
        modell.fit(x_cat_train, y_cat_train)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            st.title("Confusion Matrix")


        c1, c2 = st.columns(2)
        with c2:

            plot_confusion_matrix()
        with c1:

            input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

            if st.button('Predict', key='predict_button1'):
                predicted_quality = model_quality.predict(input_data)[0]
                if predicted_quality >= 10:
                    st.write("The quality is good")
                    st.image("good.png", width=400)
                else:
                    st.write("The quality is bad")
                    st.image("bad.png", width=400)

                accuracy = accuracy_score(y_cat_test, y_cat_pred)
                st.markdown(f"### Prediction Accuracy: {accuracy * 100:.2f}%")

    if model_type == "Random Forest Classifier":
        modell = RandomForestClassifier()
        modell.fit(x_cat_train, y_cat_train)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            st.title("Confusion Matrix")


        c1, c2 = st.columns(2)
        with c2:

            plot_confusion_matrix()
        with c1:

            input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

            if st.button('Predict', key='predict_button1'):
                predicted_quality = model_quality.predict(input_data)[0]
                if predicted_quality >= 10:
                    st.write("The quality is good")
                    st.image("good.png", width=400)
                else:
                    st.write("The quality is bad")
                    st.image("bad.png", width=400)

                accuracy = accuracy_score(y_cat_test, y_cat_pred)
                st.markdown(f"### Prediction Accuracy: {accuracy * 100:.2f}%")

    if model_type == "Decision Tree Classifier":
        modell = DecisionTreeClassifier()
        modell.fit(x_cat_train, y_cat_train)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            st.title("Confusion Matrix")


        c1, c2 = st.columns(2)
        with c2:

            plot_confusion_matrix()
        with c1:



            input_data = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]).reshape(1, -1)

            if st.button('Predict', key='predict_button1'):
                predicted_quality = model_quality.predict(input_data)[0]
                if predicted_quality >= 10:
                    st.write("The quality is good")
                    st.image("good.png", width=400)
                else:
                    st.write("The quality is bad")
                    st.image("bad.png", width=400)

                accuracy = accuracy_score(y_cat_test, y_cat_pred)
                st.markdown(f"### Prediction Accuracy: {accuracy * 100:.2f}%")



with Bio_tab:
    left_column, c0, right_column = st.columns([4,1, 4])

    with left_column:
        st.markdown(
            "<p style='text-align: justify;'>"
            "Hello, I am Santosh Chhetri, pursuing my PhD under the supervision of Dr. Mohsen Zayernouri at FMATH "
            "in the department of mechanical engineering at Michigan State University. My primary passion lies in "
            "understanding materials behavior and their failures at both micro and meso scales. I specialize in "
            "employing Discrete Dislocation Dynamics (DDD) and Molecular Dynamics Simulation to meticulously analyze "
            "the evolution of defects, unraveling the intricate pathways that lead to failure, often stemming from "
            "defect bursts. My work revolves around delving deep into the mechanisms behind material defects, "
            "offering insights into how these microscopic imperfections escalate and ultimately contribute to "
            "structural failure.</p>",
            unsafe_allow_html=True
        )

    with right_column:
        image = st.image("me.jpg")
