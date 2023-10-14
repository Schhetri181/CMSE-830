#code for the website
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt



st.header("CMSE 830 - Foundation of Data Science")
st.header("Mid-Term Project - Santosh Chhetri")
sns.set()

df = pd.read_csv("drug_use_by_age.csv")

st.header("This website will help you to perform EDA and linear regression on the data available on kaggle, drug use by age")
st.header("Here is the preview of the data set")
st.dataframe(df)
column_headers = list(df.columns)

substring = 'use'

selected_columns = [col for col in df.columns if "use" in col]


df_selected = df[selected_columns]
st.title("Do you want to include drug use frequency on your analysis")

# Create radio buttons
choice = st.radio("Choose an option:", ("Yes", "No"))


if choice == "Yes":

    sd1 = st.selectbox(
    "Select a Plot",
    (
        "Age Vs drug use data",
        "Categorical Plot",
        "Correlation plot",
    )
)
    fig, ax = plt.subplots()

    if sd1 == "Age Vs drug use data":
        x_label = st.sidebar.selectbox('x axis: ', column_headers)
        y_label = st.sidebar.selectbox('y axis: ', column_headers)
        if df[x_label].dtype == 'float64' and df[y_label].dtype == 'float64':

            #sns.scatterplot(data=df, x=x_label, y=y_label, kind="reg", color="#7f1a1a")
            sns.set(style="whitegrid")
            sns.regplot(data = df,x=x_label, y=y_label, scatter=True, color="b")
            #sns.jointplot(data=df, x=x_label, y=y_label, ax=ax, kind='kde')
        else:
            st.write("Selected columns are not numeric. Please choose numeric columns for the joint plot.")


    elif sd1 == "Categorical Plot":
        x_label = st.sidebar.selectbox('x axis: ', column_headers)
        y_label = st.sidebar.selectbox('y axis: ', column_headers)
        if df[x_label].dtype == 'float64' and df[y_label].dtype == 'float64':
            sns.regplot(data=df, x=x_label, y=y_label)
        else:
            st.write("Selected columns are not numeric. Please choose numeric columns for the joint plot.")

    elif sd1 == "Correlation plot":
        numeric_columns = df.select_dtypes(include=['number'])
        sns.heatmap(numeric_columns.corr(), cmap="pink", linewidths=.2)
        plt.show()



    st.pyplot(plt.gcf())



else:

    st.header("Please select the type of EDA you want to see")
    sd1 = st.selectbox(
        "Select a Plot",
        (
        "Age Vs drug use data",
        "Categorical Plot",
        "Correlation plot",
        "Pair Plot",
        )
    )

    fig, ax = plt.subplots()


    if sd1 == "Age Vs drug use data":
        y_label = st.sidebar.selectbox('x axis: ', selected_columns)
        x_label = st.sidebar.selectbox('y axis: ', selected_columns)
        if df[x_label].dtype == 'float64' and df[y_label].dtype == 'float64':

            sns.jointplot(data=df, x=x_label, y=y_label, kind="reg", color="#7f1a1a")
            sns.jointplot(data=df, x=x_label, y=y_label, ax=ax, kind='kde')
        else:
            st.write("Selected columns are not numeric. Please choose numeric columns for the joint plot.")


    elif sd1 == "Categorical Plot":
        y_label = st.sidebar.selectbox('x axis: ', selected_columns)
        x_label = st.sidebar.selectbox('y axis: ', selected_columns)
        if df[x_label].dtype == 'float64' and df[y_label].dtype == 'float64':
            sns.regplot(data=df, x=x_label, y=y_label)
        else:
            st.write("Selected columns are not numeric. Please choose numeric columns for the joint plot.")

    elif sd1 == "Correlation plot":
        numeric_columns = df.select_dtypes(include=['number'])
        sns.heatmap(numeric_columns.corr(), cmap="pink", linewidths=.2)
        plt.show()




    st.pyplot(plt.gcf())


    fig, ax = plt.subplots()
