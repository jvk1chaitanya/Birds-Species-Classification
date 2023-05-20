from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.exception import CustomException
from src.logger import logging

class read_transform:

    @staticmethod
    def data_transformer(train_dir, val_dir, test_dir, batch_size, img_size):
        try:
            logging.info("Data transformation started")
            train_datagen = ImageDataGenerator(width_shift_range = 0.5,height_shift_range = 0.5)
            val_datagen = ImageDataGenerator()
            test_datagen = ImageDataGenerator()
            logging.info("Data reading from train directory")
            train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(img_size, img_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
            
            logging.info("Data reading from validation directory")
            
            val_data = val_datagen.flow_from_directory(directory=val_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
            
            logging.info("Data reading from test directory")
            
            test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                        target_size=(img_size, img_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
            
            logging.info("Getting the class indices")
            labels = (train_data.class_indices)
            labels = dict((v,k) for k,v in labels.items())
            
            logging.info("Data transformation completed")
            
            return train_data, val_data, test_data, labels
        
        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e)
