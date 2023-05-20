from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Dense
from tensorflow.keras.applications import EfficientNetB0
from src.exception import CustomException
from src.logger import logging

class model_loader_trainer:
    
    @staticmethod
    def load_eff_model(n_labels,img_size):
        try:
            logging.info("Loading the efficientnet model")
            base_model = EfficientNetB0(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(img_size,img_size,3),
                    pooling='max',
                    classes=n_labels
                )
            
            logging.info("modifiying the model's layers")
            
            for layer in base_model.layers:
                layer.trainable = False

            logging.info("efficientnet model is loaded")

            return base_model
        
        except Exception as e:
            logging.info("Error in loading the efficientnet model")
            raise CustomException(e)
        
    @staticmethod
    def load_custom_model(base_model,n_labels):
        try:
            logging.info("Loading the custom model")
            
            model = Sequential()
            model.add(base_model)
            model.add(Dense(2560,activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(1280,activation='relu'))
            model.add(Dense(n_labels,activation='softmax'))
            
            logging.info("custom model is loaded")
            
            return model
        
        except Exception as e:
            logging.info("Error in loading the custom model")
            raise CustomException(e)
