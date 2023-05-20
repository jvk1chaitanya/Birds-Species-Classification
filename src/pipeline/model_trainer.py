import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.image as mimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,BatchNormalization,Dense,Conv2D,GlobalAveragePooling2D,MaxPooling2D,Flatten,SpatialDropout2D
from src.exception import CustomException
from src.logger import logging
from src.components.model_loader_trainer import model_loader_trainer
from src.components.data_read_transform import read_transform

class model_trainer:

    def train_pipeline(main_path):
        try:

            # load image data
            logging.info("Loading the data")
            train_dir = os.path.join(main_path,'train')
            test_dir = os.path.join(main_path,'test')
            val_dir = os.path.join(main_path,'valid')
            
            # reading data into image data generator
            train_data, val_data, test_data, labels = read_transform.data_transformer(train_dir, val_dir, test_dir, 32, 224)

            logging.info("data loaded successfully")

            logging.info("Loading the model")

            # finding number of labels
            n_labels = len(labels)
            img_size = 224

            # loading the EfficientNetB0 model
            base_model = model_loader_trainer.load_eff_model(n_labels,img_size)

            # adding custom layers to the model
            model = model_loader_trainer.load_custom_model(base_model,n_labels)

            logging.info("model loaded successfully")

            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001) ,loss = 'categorical_crossentropy',metrics = 'accuracy')

            logging.info("Creating the model folder")

            # create a folder to save the model
            os.makedirs(os.path.join(os.getcwd(),'Model'),exist_ok = True)
            model_path = os.path.join(os.getcwd(),'Model','final_model.h5')

            # Using tensorflow's EarlyStopping to stop training if validation loss is not decreasing
            early = tf.keras.callbacks.EarlyStopping(patience = 5)

            # Using tensorflow's Model-Checkpoint to save best model having less validation loss
            modelcheck_lr = tf.keras.callbacks.ModelCheckpoint(filepath= model_path,monitor='val_loss',save_best_only = True)

            logging.info("Training the model")

            # training the model
            model_hist = model.fit(train_data,validation_data = test_data,epochs = 1,callbacks = [modelcheck_lr,early])
            logging.info("Training the model completed")

            logging.info("Evaluating the model")

            #evaluating the model
            model.load_weights(model_path)
            eval = model.evaluate(val_data)

            print("Validation loss: ",eval[0])
            print("Validation accuracy: ",eval[1])
            logging.info("Evaluating the model completed")

        except Exception as e:
            logging.info("Error in loading the model")
            raise CustomException(e)
        
if __name__ == '__main__':
    try:
        logging.info("Starting the training pipeline")
        main_path = os.path.join(os.getcwd(),'100-bird-species')
        model_trainer.train_pipeline(main_path)

    except Exception as e:
        logging.info("Error in training pipeline")
        raise CustomException(e)

