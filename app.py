#By Neerav Desai

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Create a box inside the write title
st.write('<div style="background-color: LightGray; padding: 10px;">' '<h1 style="text-align: center; color: black">Linear Regression deployment by Neerav</h1>''</div>', unsafe_allow_html=True)

st.markdown("""<h2 style='text-align: center; font-size: 18px;'> {I have trained landpriceprediction dataset from kaggle} </h2>""", unsafe_allow_html=True)
# Upload the dataset
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write(dataset.head(10))

    # Visualize the dataset
    fig, ax = plt.subplots()
    st.title("HERE IS YOUR VISUALIZATION")
    ax.scatter(dataset['land'], dataset['price'], color='black', marker='*')
    ax.set_xlabel('Land')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    # Segregate the dataset
    land = dataset.drop('price', axis='columns')
    price = dataset.price

    # Training the linear regression model
    model = linear_model.LinearRegression()
    model.fit(land, price)

    # User input for land area
    st.subheader("Enter Land Area for Prediction")
    area_of_land = st.number_input("Enter the area of land:", min_value=500)

    if area_of_land is not None:
        # Predicting the land price
        input_data = [[area_of_land]]
        predicted_price = model.predict(input_data)
        st.subheader("Predicted Price:")
        st.write(predicted_price)
    else:
        st.write("enter valid value")

