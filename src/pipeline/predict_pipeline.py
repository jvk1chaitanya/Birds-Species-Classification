import tensorflow as tf
import os
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from io import BytesIO
import numpy as np
from src.exception import CustomException
from src.logger import logging

class predict_species:
    # Load the saved model
    model = tf.keras.models.load_model('model/final_model.h5')

    # get the class names
    class_names = os.listdir(r'D:\personal-projects\birds-classification\100-bird-species\train')

    @classmethod
    def predict_pipeline(cls,image_path):
        try:
            logging.info("Starting the prediction pipeline")
            # Define the input image size
            input_shape = (224, 224)
            
            logging.info("Reading the image")
            # load image data
            img = load_img(image_path, target_size=input_shape)
            
            # convert image to array
            x = img_to_array(img)
            
            logging.info("Preprocessing the image")
            # preprocess the image using efficientnet preprocessor
            x = preprocess_input(x)

            logging.info("Making the prediction")
            # Make the prediction
            preds = cls.model.predict(np.array([x]))

            # get the index of the predicted class
            index = np.argmax(preds)

            logging.info("Getting the predicted class name")
            # get the predicted class name
            predicted_class = cls.class_names[index]

            logging.info("Predicted the class successfully")
            
            return predicted_class
        
        except Exception as e:
            logging.info("Error in predicting the class in predict_pipeline")
            raise CustomException(e)

    @classmethod
    def predict_pipeline_url(cls,image_url):
        try:
            logging.info("Starting the prediction pipeline from URL")

            logging.info("Getting the image from the url")
            # get the image from the url
            response = requests.get(image_url)

            # Define the input image size
            input_shape = (224, 224)
            
            logging.info("Reading the image")
            # load image data
            img = load_img(BytesIO(response.content), target_size=input_shape)
            
            # convert image to array
            x = img_to_array(img)
            
            logging.info("Preprocessing the image")
            # preprocess the image using efficientnet preprocessor
            x = preprocess_input(x)

            logging.info("Making the prediction")
            # Make the prediction
            preds = cls.model.predict(np.array([x]))

            # get the index of the predicted class
            index = np.argmax(preds)

            logging.info("Getting the predicted class name")
            # get the predicted class name
            predicted_class = cls.class_names[index]

            logging.info("Predicted the class successfully")
            
            return predicted_class
        
        except Exception as e:
            logging.info("Error in predicting the class in predict_pipeline")
            raise CustomException(e)

"""        
if __name__ == '__main__':
    try:
        print(predict_species.predict_pipeline_url(r'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/78541101/1200'))

    except CustomException as e:
        logging.info("Error in predicting the class")
        raise CustomException(e)
"""