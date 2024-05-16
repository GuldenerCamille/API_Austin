import streamlit as st
import pandas as pd
import requests
import joblib
import json
from utils import *

# model = joblib.load("src/assets/model.pkl") #CHANGE TO MODEL PATH

if "reset" not in st.session_state:
    st.session_state["reset"] = False

col1, col2 = st.columns((1, 1), gap="small")

with col1:
    st.markdown("#")
    st.markdown("#")
    with st.container(border=True):
        header("Animal adoption", 3)
        header("Enter the animal's information :", 4)



        col11, col12 = st.columns((1, 1), gap="medium")

        with col11:

            Intake_Type = st.text_input(label="**Intake Type :**")
            Intake_Condition = st.text_input(label="**Intake Condition :**")
            Animal_Type = st.text_input(label="**Animal Type :**")

        with col12:
            Sex_upon_Intake = st.text_input(label="**Sex upon Intake :**")
            Age_Upon_Intake_Days = st.number_input(label="**Age in days :**", step = 1)
            Animal_Group = st.number_input(label="**Animal Group :**", step = 1)
            status_validated = st.text_input(label="**status_validated**")

        model_data_dict = {
            "Intake_Type": Intake_Type,
            "Intake_Condition": Intake_Condition,
            "Animal_Type" : Animal_Type,
            "Sex_upon_Intake": Sex_upon_Intake,
            "Animal_Group": Animal_Group,
            "Age_Upon_Intake_Days": Age_Upon_Intake_Days
        }

        model_data_df = pd.DataFrame(model_data_dict, index=[0])

# Add a button for prediction
if st.button("Get Prediction"):
    # Create a dictionary with user inputs
    model_data_dict = {
        "Intake_Type": Intake_Type,
        "Intake_Condition": Intake_Condition,
        "Animal_Type": Animal_Type,
        "Sex_upon_Intake": Sex_upon_Intake,
        "Animal Group": Animal_Group,
        "Age_Upon_Intake_Days": Age_Upon_Intake_Days
    }

    # Send the data to the API for prediction
    response = requests.get(f'https://projete1-api.onrender.com/predict?Intake_Type={model_data_dict["Intake_Type"]}&Intake_Condition={model_data_dict["Intake_Condition"]}&Animal_Type={model_data_dict["Animal_Type"]}&Sex_upon_Intake={model_data_dict["Sex_upon_Intake"]}&Age_Upon_Intake_Days={model_data_dict["Age_Upon_Intake_Days"]}')


    # Check if the request was successful
    if response.status_code == 200:
        st.success("Prediction received successfully!")
        prediction = response.json()  # Assuming API returns JSON
        # Display the prediction or do something with it
        st.write(prediction)
    else:
        st.error(f"Failed to get prediction. Error code: {response.status_code}")


    # model_data_df = pd.DataFrame(model_data_dict, index=[0])

    # response = requests.get(f'https://projete1-api.onrender.com/predict?Intake_Type={model_data_dict["Intake_Type"]}&Intake_Condition={model_data_dict["Intake_Condition"]}&Animal_Type={model_data_dict["Animal_Type"]}&Sex_upon_Intake={model_data_dict["Sex_upon_Intake"]}&Age_Upon_Intake_Days={model_data_dict["Age_Upon_Intake_Days"]}')
    # model_data_df["Prediction"]=json.loads(response.content.decode("utf-8"))["result"][0]
