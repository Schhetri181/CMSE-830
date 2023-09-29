#code for the website
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt



st.header("CMSE 830")
st.header("Hello this is santosh Chhetri, this is my website")
sns.set()

df = pd.read_csv("C:/Users/chhet/Downloads/CMSE 830/drug_use_by_age.csv")


st.header("Here is the preview of the drug use by age data set")
st.dataframe(df)
st.header("Please select the type of EDA you want to see")
sd1 = st.selectbox(
    "Select a Plot",
    (
        "Joint plot",
        "Categorical Plot",
        "Correlation plot",
        "Pair Plot",
    )
)
column_headers = list(df.columns)
dd = ['Number of drivers involved in fatal collisions per billion miles',
       'Percentage Of Drivers Involved In Fatal Collisions Who Were Speeding',
       'Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired',
       'Percentage Of Drivers Involved In Fatal Collisions Who Were Not Distracted',
       'Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents',
       'Car Insurance Premiums ($)',
       'Losses incurred by insurance companies for collisions per insured driver ($)']
x_label = st.sidebar.selectbox('x axis: ', column_headers)
y_label = st.sidebar.selectbox('y axis: ', column_headers)



fig, ax = plt.subplots()


if sd1 == "Joint plot":
    if df[x_label].dtype == 'float64' and df[y_label].dtype == 'float64':

        sns.jointplot(data=df, x=x_label, y=y_label, kind="reg", color="#7f1a1a")
        sns.jointplot(data=df, x=x_label, y=y_label, ax=ax, kind='kde')
    else:
        st.write("Selected columns are not numeric. Please choose numeric columns for the joint plot.")


elif sd1 == "Categorical Plot":
    if df[x_label].dtype == 'float64' and df[y_label].dtype == 'float64':
        sns.regplot(data=df, x=x_label, y=y_label)
    else:
        st.write("Selected columns are not numeric. Please choose numeric columns for the joint plot.")

elif sd1 == "Correlation plot":
    numeric_columns = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_columns.corr(), cmap="pink", linewidths=.2)
    plt.show()

elif sd1 =="Pair Plot":
    sns.pairplot(data = df)


st.pyplot(plt.gcf())