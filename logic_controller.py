import tensorflow as tf
import numpy as np
import json
import os


class Data_controller:
    def __init__(self):
        self.model = tf.keras.models.load_model('model/trained_model.h5')

    def model_prediction(self,data_image):
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

        image = tf.keras.preprocessing.image.load_img(data_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        prediction = self.model.predict(input_arr)
        result_index = np.argmax(prediction)

        plant_name_mapping = {
            # Apple diseases
            'Apple___Apple_scab': 'Apple - Apple Scab',
            'Apple___Black_rot': 'Apple - Black Rot',
            'Apple___Cedar_apple_rust': 'Apple - Cedar Apple Rust',
            'Apple___healthy': 'Healthy Apple',

            # Blueberry
            'Blueberry___healthy': 'Healthy Blueberry',

            # Cherry diseases
            'Cherry_(including_sour)___Powdery_mildew': 'Cherry (including sour) - Powdery Mildew',
            'Cherry_(including_sour)___healthy': 'Healthy Cherry',

            # Corn (maize) diseases
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn (Maize) - Cercospora Leaf Spot/Gray Leaf Spot',
            'Corn_(maize)___Common_rust_': 'Corn (Maize) - Common Rust',
            'Corn_(maize)___Northern_Leaf_Blight': 'Corn (Maize) - Northern Leaf Blight',
            'Corn_(maize)___healthy': 'Healthy Corn (Maize)',

            # Grape diseases
            'Grape___Black_rot': 'Grape - Black Rot',
            'Grape___Esca_(Black_Measles)': 'Grape - Esca (Black Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape - Leaf Blight (Isariopsis Leaf Spot)',
            'Grape___healthy': 'Healthy Grape',

            # Orange diseases
            'Orange___Haunglongbing_(Citrus_greening)': 'Orange - Huanglongbing (Citrus Greening)',

            # Peach diseases
            'Peach___Bacterial_spot': 'Peach - Bacterial Spot',
            'Peach___healthy': 'Healthy Peach',

            # Pepper diseases
            'Pepper,_bell___Bacterial_spot': 'Bell Pepper - Bacterial Spot',
            'Pepper,_bell___healthy': 'Healthy Bell Pepper',

            # Potato diseases
            'Potato___Early_blight': 'Potato - Early Blight',
            'Potato___Late_blight': 'Potato - Late Blight',
            'Potato___healthy': 'Healthy Potato',

            # Raspberry
            'Raspberry___healthy': 'Healthy Raspberry',

            # Soybean
            'Soybean___healthy': 'Healthy Soybean',

            # Squash diseases
            'Squash___Powdery_mildew': 'Squash - Powdery Mildew',

            # Strawberry diseases
            'Strawberry___Leaf_scorch': 'Strawberry - Leaf Scorch',
            'Strawberry___healthy': 'Healthy Strawberry',

            # Tomato diseases
            'Tomato___Bacterial_spot': 'Tomato - Bacterial Spot',
            'Tomato___Early_blight': 'Tomato - Early Blight',
            'Tomato___Late_blight': 'Tomato - Late Blight',
            'Tomato___Leaf_Mold': 'Tomato - Leaf Mold',
            'Tomato___Septoria_leaf_spot': 'Tomato - Septoria Leaf Spot',
            'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato - Spider Mites/Two-spotted Spider Mite',
            'Tomato___Target_Spot': 'Tomato - Target Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato - Yellow Leaf Curl Virus',
            'Tomato___Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
            'Tomato___healthy': 'Healthy Tomato'
        }

        return class_name[result_index]

    def recommendation_management(self,disease_name):
        f = open('model/data.json')
        data = json.load(f)

        Nutrient_Management_Soil = data[disease_name]['Nutrient_Management']['Soil_pH']
        Nutrient_Management_Nitrogen = data[disease_name]['Nutrient_Management']['Nitrogen']
        Nutrient_Management_Phosphorus = data[disease_name]['Nutrient_Management']['Phosphorus']
        Nutrient_Management_Potassium = data[disease_name]['Nutrient_Management']['Potassium']

        Fertilizer_Recommendation = data[disease_name]['Fertilizer_Recommendation']
        Pest_Control = data[disease_name]['Pest_Control']
        Weather_Considerations = data[disease_name]['Weather_Considerations']
        Preventive_Measures = data[disease_name]['Preventive_Measures']

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

        return recommendation_output

    def plant_name_mapping(self,plant_name):

        plant_name_mapping = {
            # Apple diseases
            'Apple___Apple_scab': 'Apple - Apple Scab',
            'Apple___Black_rot': 'Apple - Black Rot',
            'Apple___Cedar_apple_rust': 'Apple - Cedar Apple Rust',
            'Apple___healthy': 'Healthy Apple',

            # Blueberry
            'Blueberry___healthy': 'Healthy Blueberry',

            # Cherry diseases
            'Cherry_(including_sour)___Powdery_mildew': 'Cherry (including sour) - Powdery Mildew',
            'Cherry_(including_sour)___healthy': 'Healthy Cherry',

            # Corn (maize) diseases
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn (Maize) - Cercospora Leaf Spot/Gray Leaf Spot',
            'Corn_(maize)___Common_rust_': 'Corn (Maize) - Common Rust',
            'Corn_(maize)___Northern_Leaf_Blight': 'Corn (Maize) - Northern Leaf Blight',
            'Corn_(maize)___healthy': 'Healthy Corn (Maize)',

            # Grape diseases
            'Grape___Black_rot': 'Grape - Black Rot',
            'Grape___Esca_(Black_Measles)': 'Grape - Esca (Black Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape - Leaf Blight (Isariopsis Leaf Spot)',
            'Grape___healthy': 'Healthy Grape',

            # Orange diseases
            'Orange___Haunglongbing_(Citrus_greening)': 'Orange - Huanglongbing (Citrus Greening)',

            # Peach diseases
            'Peach___Bacterial_spot': 'Peach - Bacterial Spot',
            'Peach___healthy': 'Healthy Peach',

            # Pepper diseases
            'Pepper,_bell___Bacterial_spot': 'Bell Pepper - Bacterial Spot',
            'Pepper,_bell___healthy': 'Healthy Bell Pepper',

            # Potato diseases
            'Potato___Early_blight': 'Potato - Early Blight',
            'Potato___Late_blight': 'Potato - Late Blight',
            'Potato___healthy': 'Healthy Potato',

            # Raspberry
            'Raspberry___healthy': 'Healthy Raspberry',

            # Soybean
            'Soybean___healthy': 'Healthy Soybean',

            # Squash diseases
            'Squash___Powdery_mildew': 'Squash - Powdery Mildew',

            # Strawberry diseases
            'Strawberry___Leaf_scorch': 'Strawberry - Leaf Scorch',
            'Strawberry___healthy': 'Healthy Strawberry',

            # Tomato diseases
            'Tomato___Bacterial_spot': 'Tomato - Bacterial Spot',
            'Tomato___Early_blight': 'Tomato - Early Blight',
            'Tomato___Late_blight': 'Tomato - Late Blight',
            'Tomato___Leaf_Mold': 'Tomato - Leaf Mold',
            'Tomato___Septoria_leaf_spot': 'Tomato - Septoria Leaf Spot',
            'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato - Spider Mites/Two-spotted Spider Mite',
            'Tomato___Target_Spot': 'Tomato - Target Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato - Yellow Leaf Curl Virus',
            'Tomato___Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
            'Tomato___healthy': 'Healthy Tomato'
        }


        readable_name = plant_name_mapping.get(plant_name, plant_name)

        return readable_name

    def fix_file_path(self,filepath):

        # Normalize the path (converts to OS-specific slashes)
        normalized_path = os.path.normpath(filepath)
        # Convert to forward slashes
        forward_slash_path = normalized_path.replace('\\', '/')
        return forward_slash_path













