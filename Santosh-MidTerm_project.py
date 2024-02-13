# code for the website

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import hiplot as hip

st.title("Data Visualization on Drug use by different age group")

#st.markdown("### Mid-Term Project - Santosh Chhetri", unsafe_allow_html=True)

image = "drug-use.png"
st.image(image, caption="Picture credit- Internet", use_column_width=False, width=900)

# st.write(
 #   'Drug use seems like a consistent problem all across the US with no end in sight. Its a constant point of note'
 #   'in daily news.Substance misuse, including the abuse of prescription medications, illicit drugs, and alcohol, '
 #   'is widespread and has far-reaching consequences. Health issues, addiction, overdose, and mental health disorders'
 #   ' are common outcomes of drug use. However, there is a lack of proper analysis to understand the extent of the '
 #   'problem where it affects people the worst. Through this project, I try to understand drug use by different '
 #   'demographics especially age. Through a thorough understanding of how drug use is different across various '
 #   'demographics we can create targeted programs to support people through their troubles of drug use.'
# )

df = pd.read_csv("drug_use_by_age.csv")

st.markdown("##### This website will help you to perform EDA using different visualisation tools available in seaborn, "
            ", altair, and hiplot.")

st.markdown("##### Would you like to hide the preview of the data set")
st.write("")
data_choice = st.radio("Please select your option", ("Yes", "No"), key="data_choice")
if data_choice == "Yes":
    st.markdown("#### Data is hidden")

else:
    st.write("Here is the preview of the data set")
    st.dataframe(df)
column_headers = list(df.columns)

substring = 'use'
selected_columns = [col for col in df.columns if "use" in col]
df_selected = df[selected_columns]

st.sidebar.markdown("#### Do you want to include drug use frequency on your analysis")

choice = st.sidebar.radio("Choose an option:", ("Yes", "No"))

st.sidebar.markdown("#### Please select the type of EDA you want to perform")
sd1 = st.sidebar.selectbox(
    "Select a Plot",
    (
        "Age Vs drug use data",
        "Regression plot",
        "Correlation plot",
        "Distribution plot - Violin and box plots",
        "Distribution plot - KDE",
    )
)

if sd1 == "Age Vs drug use data":
    st.markdown("#### Age Vs drug use")

   # st.write(
   #     "This plot will help to visualize the trend in drug usage with respect to age. Most of the drug use"
   #     " often begins in the teenage year, with many adolescents trying different things in life. Almost all"
   #     " of the plots shows that any kind of drug use is at its peak in their 20s. However, the drug use"
   #     " seems to start declining after early 20s. From this plot we can infer that almost any kind of "
   #     "drug use start at their teenage be it by their curiosity or experiments. But, there is gradual decrease"
   #     " in drug use after late 20s and that might be due to the responsibilities in later life."
   # )

    x_label = "age"
    if choice == 'Yes':
        column_headers = column_headers[2:-1]
    else:
        column_headers = selected_columns[:-1]

    y_label = st.sidebar.selectbox('y axis: ', column_headers)
    alt_chart = alt.Chart(df).mark_point().encode(
        x=x_label,
        y=y_label,
        tooltip=[x_label, y_label],
    ).interactive()
    alt_chart = alt_chart.configure_axis(gridOpacity=0.75)
    st.altair_chart(alt_chart, use_container_width=True)

elif sd1 == "Regression plot":
#st.write(
#        "The regression  provides a convenient way to visualize the linear "
#        "relationship between two variables. It combines a scatterplot showing the individual data points with a "
#        "regression line displaying the fitted linear model. Regression fits a simple linear regression and plots "
#        "the resulting line along with a 95% confidence interval band to capture the uncertainty around the "
#        "estimate. This allows understanding the correlation and potential predictive relationship"
#    )
    if choice == 'Yes':
        column_headers = column_headers[2:-1]
    else:
        column_headers = selected_columns
    x_label = st.sidebar.selectbox('x axis: ', column_headers)
    y_label = st.sidebar.selectbox('y axis: ', column_headers)

    sns.regplot(data=df, x=x_label, y=y_label)
elif sd1 == "Correlation plot":
  
#st.write(
#        'Seaborn\'s heatmap function is a powerful data visualization tool for representing complex datasets as '
#        'color-coded matrices. It is particularly useful for visualizing the relationships and patterns within '
#        'a dataset, making it easier to spot trends or correlations. Here, dark red color means the strong '
#       'positive correlations, and strong blue means the strong negative correlation between these datasets.'
#        'Apart from the diagonal, marijuana shows strong positive correlation with hallucinogen, pain-reliever,'
#        'oxytocin and stimulant use.'
 #   )

    df1 = df
    df1 = df1.drop(columns=['n', 'age', 'Age-group-use'])
    if choice == 'Yes':
        numeric_columns = df1.select_dtypes(include=['int64', 'float64'])
        column_headers = df1.select_dtypes(include=['int64', 'float64'])
    else:

        column_headers = selected_columns[:-1]
        column_headers = df[column_headers]
        numeric_columns = column_headers

    sns.heatmap(numeric_columns.corr(), cmap="coolwarm", linewidth=0.5)
    plt.xticks(range(len(numeric_columns.columns)), numeric_columns.columns)
    show_correlation_data = st.checkbox("Show Correlation Data", value=False)
    if show_correlation_data:
        st.dataframe(numeric_columns.corr())
    else:
        st.markdown("### Correlation data is hidden")


elif sd1 == 'Distribution plot - Violin and box plots':
    st.markdown("### Violin plot")
    
st.write(
        'The violin plot is a combination of a box plot and a kernel density plot. It displays the distribution of '
        'data across different categories or groups, allowing you to compare and contrast their shapes and '
        'central tendencies. The violin shape represents the estimated probability density of the data at '
        'different values, while the box inside the violin shows the inter-quartile range (IQR) of the data, which'
        ' includes the median (center line). '
    )
    
    st.markdown("### Box Plot")
 
st.write(
        'Box plot, also known as a box-and-whisker plot, provides a more concise summary of the data '
        'distribution. It consists of a rectangular box that spans the IQR and whiskers that extend to the '
        'minimum and maximum values within a defined range (typically 1.5 times the IQR). Box plots are effective'
        ' in identifying the central tendency, spread, and any potential outliers in the data.'
    )
    
    y_label = st.sidebar.selectbox('Select the dataset to see the plots: ', selected_columns[:-1])
    custom_palette={"teens": "blue", "twentees": "green", "middle_age":"yellow", "senior_age":"red"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    sns.violinplot(data=df, y=y_label, x="Age-group-use", ax=ax1, palette=custom_palette)
    ax1.set_title("Violin Plot")
    sns.boxplot(data=df, y=y_label, x="Age-group-use", ax=ax2,palette=custom_palette)
    ax2.set_title("Box Plot")
    plt.subplots_adjust(hspace=0.5)
elif sd1 == "Distribution plot - KDE":
    st.markdown("### KDE")

    st.write(
        'The kernel density estimation (KDE) is a non-parametric method to estimate the probability density '
        'function of a random variable. KDE is done by placing a kernel function (eg- gaussian) at each data point'
        'and adding them together to form a smooth curve. uses kernel density estimation to generate a smooth '
        'curve based on the data. Here, dis-plot function from seaborn is used and customized to add multiple kde'
        ' curves for comparison. Choice of bandwidth plays an important rule in the overall shape of the KDE, a'
        ' larger bandwidth leads to the smoother curve while narrow bandwidth results in spikes in the final plot.'
    )
    
    x_label = st.sidebar.selectbox('Select the dataset to see the plots: ', selected_columns[:-1])
    width = st.slider('Adjust Bandwidth:', min_value=0.1, max_value=10.0, value=4.0, step=0.1)
    sns.displot(data=df, x=x_label, hue="Age-group-use", kind="kde", bw_adjust=width)

st.pyplot(plt.gcf())
st.markdown('### HiPlot')

st.write(
    'HiPlot is an interactive visualization tool designed for exploring relationships in high-dimensional data. It '
    'utilizes a combination of parallel coordinates and scatter plot matrices to provide a multivariate view of '
    'complex data. With parallel coordinates, each vertical axis represents one dimension or variable from the dataset.'
    ' Lines are then drawn connecting each data point across all axes, allowing you to see correlations between the '
    'different dimensions.'
)

st.header("Do you want to see hi-plot")
hiplot_choice = st.radio("Choose an option:", ("Yes", "No"), key="hiplot_choice")

if hiplot_choice == "Yes":
    experiment = hip.Experiment.from_dataframe(df_selected)
    st.components.v1.html(experiment.to_html(), width=800, height=600, scrolling=True)
else:
    st.write("HiPlot is hidden")
