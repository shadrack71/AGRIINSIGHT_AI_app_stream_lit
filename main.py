import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


import streamlit as st
import json


def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size = (128,128))
    input_arr= tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def recommendation_management(disease_name):
    f = open('data.json')
    data = json.load(f)
    
    return data[disease_name]
    
    
    
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Dataset","Team","AGRIINSIGHT Dashboard"])

#Main Page
if(app_mode=="Home"):
    st.header(" AN AI-POWERED CROP HEALTH AND MANAGEMENT SYSTEM FOR EARLY DISEASE DETECTION, PRECISION NUTRIENT OPTIMIZATION, AND SMART PEST CONTROL.")
    image_path = "image.png"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    #### Welcome to AGRIINSIGHT: Your Farming Partner for Optimal Growth! 🌱📊🌿🔍
    AGRIINSIGHT is an innovative app designed to help farmers optimize fertilizer usage and ensure plant health. By analyzing soil conditions, crop type, and local weather data, we provide personalized fertilizer recommendations while also identifying potential plant diseases—empowering farmers to increase productivity while maintaining environmental sustainability.Our mission is to help in identifying plant diseases efficiently.
    Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    
    ### How AGRIINSIGHT Works:
    ##### 1.**Input Data:** Upload images of your plants  and the soil where it is grown.
    ##### 2.**Smart Analysis:** AGRIINSIGHT’s advanced algorithms analyze your input to recommend the ideal fertilizer usage and detect plant diseases.
    ##### 3.**Instant Recommendations:** Receive actionable insights and suggestions to enhance crop yields and ensure sustainable farming practices.
   
    
    """)
    image_path = "mission.png"
    st.image(image_path,use_container_width=True)
    image_path = "vision.png"
    st.image(image_path,use_container_width=True)
    
    image_path = "problem.png"
    st.image(image_path,use_container_width=True)
    st.markdown("""
   
    ### Why Choose AGRIINSIGHT:
    ##### 1. **Data-Driven Precision:** Leverages the latest machine learning and data analytics for accurate recommendations tailored to your farm's needs.
    ##### 2. **User-Friendly Interface:** Designed for farmers of all experience levels with an intuitive and easy-to-navigate layout.
    ##### 3. **Quick, Impactful Results:** Get fast, reliable insights that help you make timely decisions for healthier crops and a reduced environmental footprint.


    ### Get Started
    CVisit the AGRIINSIGHT Dashboard to upload your data and take your farm’s productivity to the next level with precision and sustainability.

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
    
    image_path = "system.png"
    st.image(image_path,use_container_width=True)
    
#About Project
elif(app_mode=="Dataset"):
    st.header("About Our Dataset")
    st.markdown("""
                #### About Our Dataset 📊

                The dataset powering AGRIINSIGHT was carefully recreated through offline augmentation from the original dataset, which can be found in the associated GitHub repo. It contains approximately 87,000 RGB images of both healthy and diseased crop leaves, categorized into 38 different plant disease classes
                The dataset is split into training and validation sets in an 80/20 ratio while preserving the directory structure for easier management. Additionally, a separate directory with 33 test images was created to assist with prediction tasks.

                #### Dataset Breakdown:
                1. Train Set (1000 images)
                2. Test Set(20 images)
                3. Validation Set (200 images)

                """)
    
    
    image_path = "data.png"
    st.image(image_path,use_container_width=True)
    image_path = "technical.png"
    st.image(image_path,use_container_width=True)
    image_path = "machine.png"
    st.image(image_path,use_container_width=True)
    
elif(app_mode=="Team"):
    st.header("About Our Team")
    st.markdown("""
                ## About Our Team 📊
                
                At AGRIINSIGHT, our team is composed of passionate individuals from diverse fields such as agriculture, data science, and software development. Together, we work to leverage cutting-edge technologies to provide farmers with actionable insights that improve their productivity and support sustainable farming practices.

                Our team is committed to constantly improving the system by integrating new features, enhancing algorithm accuracy, and expanding the dataset for even better disease detection and fertilizer optimization. We aim to bridge the gap between technology and agriculture, ensuring farmers have the tools they need to thrive in a changing world.
                
                
                
                
                ##Our Commitment 🌍
                At AGRIINSIGHT, we believe in the power of technology to create a sustainable future for agriculture. Our commitment lies in:

                **Accuracy:** Using state-of-the-art algorithms to provide farmers with the most accurate and actionable insights.
                **Sustainability:** Promoting environmentally-friendly farming practices by recommending optimal fertilizer usage and reducing unnecessary chemical **application.
                **Empowerment:** Equipping farmers with tools that enhance their decision-making process and enable them to take control of their farm’s health and productivity.
                We are dedicated to providing a reliable, user-friendly experience for farmers and ensuring that our app delivers meaningful results that can positively impact global agricultural practices.
                
                
                """)
#Prediction Page
elif(app_mode=="AGRIINSIGHT Dashboard"):
    st.header("AGRIINSIGHT: Smart Farming for Sustainable Agriculture")
    st.subheader(" AGRIINSIGHT AI is an AI-powered platform that integrates Disease Recognition")
    image_path = "work.png"
    st.image(image_path,use_container_width=True)
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("AGRIINSIGHT AI is an AI-powered platform")
        result_index = model_prediction(test_image)
        
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("AGRIINSIGHT AI Results it's a {}".format(class_name[result_index]))
        data_result =  recommendation_management(class_name[result_index])
        
        Nutrient_Management_Soil = data_result['Nutrient_Management']['Soil_pH']
        Nutrient_Management_Nitrogen = data_result['Nutrient_Management']['Nitrogen']
        Nutrient_Management_Phosphorus = data_result['Nutrient_Management']['Phosphorus']
        Nutrient_Management_Potassium = data_result['Nutrient_Management']['Potassium']
        
        Fertilizer_Recommendation = data_result['Fertilizer_Recommendation']
        Pest_Control = data_result['Pest_Control']
        Weather_Considerations = data_result['Weather_Considerations']
        Preventive_Measures = data_result['Preventive_Measures']
        
        recommendation_output = (
                f"Nutrient Management:\n"
                f"  - Soil pH: {Nutrient_Management_Soil}\n"
                f"  - Nitrogen: {Nutrient_Management_Nitrogen}\n"
                f"  - Phosphorus: {Nutrient_Management_Phosphorus}\n"
                f"  - Potassium: {Nutrient_Management_Potassium}\n\n"
                f"Fertilizer Recommendation:\n"
                f"  - {Fertilizer_Recommendation}\n\n"
                f"Pest Control:\n"
                f"  - {Pest_Control}\n\n"
                f"Weather Considerations:\n"
                f"  - {Weather_Considerations}\n\n"
                f"Preventive Measures:\n"
                f"  - {Preventive_Measures}"
            )

        

        st.success(recommendation_output)
       
