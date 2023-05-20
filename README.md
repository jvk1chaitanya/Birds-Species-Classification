# Bird Species Recognition Using Efficient-Net
The goal of this project is to build a Bird Recognition model using EfficientNetB0, a highly accurate and efficient neural network architecture designed specifically for image classification tasks. With this model, we aim to accurately predict the 525 species of birds found in our dataset.

This project has practical applications in a range of fields, including wild-life monitoring, chat-bot integration, and more. By accurately identifying different bird species, we can gain a better understanding of bird populations, track migration patterns, and support conservation efforts.

To train and test our model, we will be using a large dataset of bird images. This dataset includes high-quality images of birds in a range of poses, backgrounds, and lighting conditions, providing a diverse set of training examples for our model.

## About the Dataset
The dataset used in this project is a directory that contains 525 different species of bird images, with each species group into a separate folder. The dataset is available on Kaggle and can be downloaded from [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

The dataset provides a diverse range of high-quality bird images that have been collected from a variety of sources, including birdwatching websites, bird identification guides, and nature photographers.

All images in the dataset have been standardized to a resolution of 224 x 224 pixels, which is the input size required by the EfficientNetB0 architecture used in this project. By providing a large and diverse set of training examples, this dataset allows us to build a highly accurate bird recognition model capable of identifying a wide range of bird species in the wild.

## Python Libraries Used

In this project, we have used various Python libraries for building and training our deep neural network model. Here's a brief overview of each library, organized by their purpose:

### Deep Learning Frameworks and Tools
- tensorflow: An open-source machine learning framework developed by Google. It is widely used for building and training deep neural networks.

### Operating System Interaction and Data Handling
- os: A module in Python that provides a way of interacting with the operating system. It allows you to work with directories, files, and other OS-related tasks.
- random: A module in Python that provides functions to generate pseudo-random numbers.

### Data Visualization
- matplotlib: A plotting library for Python. It provides a range of useful visualizations, such as line plots, scatter plots, bar plots, and histograms.
- seaborn: A data visualization library built on top of matplotlib. It provides a higher-level interface for creating more complex visualizations.

### Image Data Handling and Preprocessing
- ImageDataGenerator: A class in the TensorFlow Keras API used for generating batches of image data. It can perform data augmentation techniques such as flipping, rotating, and zooming on the fly.
- load_img: A function in the TensorFlow Keras API used to load an image from a file.
- img_to_array: A function in the TensorFlow Keras API used to convert an image to a numpy array.
- preprocess_input: A function in the TensorFlow Keras API used to preprocess input data for specific models. In this case, it is used to preprocess input images for the EfficientNetB0 model.

### Model Building and Training
- Sequential: A class in the TensorFlow Keras API used for building sequential models. It allows you to create a model by simply adding layers one after another.
- Dense: A class in the TensorFlow Keras API used for adding fully connected layers to a neural network.
- BatchNormalization: A technique used to improve the training speed and stability of deep neural networks.
- EfficientNetB0: A pre-trained deep learning model for image classification. It is part of the EfficientNet family of models, which are known for their high accuracy and efficiency.

### HTTP Request Handling
- Requests: A Python library that allows you to send HTTP/1.1 requests using Python, making it easy to interact with web services and access web resources.

### Input and Output Handling
- io: The 'io' module in Python provides a way to handle various types of I/O operations such as reading/writing data to/from files, bytes, strings, and other types of objects.

By using these libraries, we can build a highly accurate and efficient bird recognition model capable of identifying a wide range of bird species.

## Data Preprocessing and Categorization
Before we start using images to train our model, we need to preprocess and categorize the data. In this project, we are using the `ImageDataGenerator` class from the TensorFlow Keras API to perform data preprocessing and categorization.

The `ImageDataGenerator` class allows us to generate batches of image data and perform various data augmentation techniques such as flipping, rotating, and zooming on the fly. It also provides an easy way to categorize the data based on the folder structure of the dataset.

To categorize the data, we have organized the images into folders based on their respective species. We have used the `flow_from_directory` method of the `ImageDataGenerator` class to read the images from the directories and categorize them based on the folder structure.

We have also used the `class_mode` parameter of the `flow_from_directory` method to specify that the data should be categorized into multiple classes. In our case, we have set `class_mode` to `'categorical'`, which means that the labels are one-hot encoded.

Overall, the data preprocessing and categorization steps are crucial for building an accurate and efficient bird species recognition model.

## Model Architecture

For our model, we are using transfer learning technique.

"Transfer learning is a technique in machine learning and artificial intelligence that involves taking knowledge learned from one task and applying it to a different but related task. In other words, transfer learning allows a model that has been trained on a specific task to leverage the knowledge it has gained to perform a different but related task with greater efficiency or accuracy. This is often done by using a pre-trained model, which has already been trained on a large dataset, as a starting point for training a new model for a different task. Transfer learning is particularly useful in cases where the new task has limited training data, as it allows the model to generalize better from the pre-existing knowledge."

After reading the above definition of transfer learning, you will observe that there is a lot of mention of a pre-trained model, and you might want to know which pre-trained model we are using. Well, in our case, we are using EfficientNetB0, which is a part of the Efficient-Net family. EfficientNet has seven different variations, which are typically denoted as EfficientNet-B0, EfficientNet-B1, EfficientNet-B2, EfficientNet-B3, EfficientNet-B4, EfficientNet-B5, and EfficientNet-B6. In comparison, EfficientNet-B6 is more complex and provides better results than that of EfficientNet-B0. 

Then why are we using EfficientNet-B0 instead of EfficientNet-B6? It's because higher the model complexity, more time and resources are required to train the model.

After picking our pre-trained model, we will use the weights of "imagenet" to set weights in our pre-trained model layers, and we will freeze these layers so that the weights don't change while we are training our model. We are doing this because the pre-trained model is already good at generalizing the image data, as it was trained on a larger set of data.

You might also see that while fetching the EfficientNet-B0 model through API, we have set TOP_layer as False. This is because we will make the top layer of our own, which will make our model generalize on bird species accurately. To the existing pre-trained model, we are adding a series of these layers:

1) Dense layer with activation relu - A dense layer connects every neuron in one layer to every neuron in the next layer, while the activation function relu introduces non-linearity by setting negative values to zero, enabling the model to learn complex patterns.
2) BatchNormalization - Batch normalization is a technique used in deep learning to normalize the inputs of each layer, reducing the internal covariate shift and improving the overall performance and stability of the network.
3) Dense layer with activation softmax - In a neural network, a dense layer followed by the activation function softmax is used to produce a probability distribution over the classes of a classification task.

## Model Training
Training a machine learning model requires careful selection of hyperparameters and optimization techniques to achieve optimal performance. For our bird species recognition model, we employed the Adam optimizer and categorical cross-entropy loss to train the model.

To avoid overfitting, we employed two techniques: model checkpoint and early stopping. The model checkpoint allowed us to save the best model during training, enabling us to resume training from the best model if the training process was interrupted. On the other hand, early stopping stopped the training process if the model's performance did not improve after a certain number of epochs, thereby preventing the model from overfitting to the training data.

Our model was trained using a large dataset of bird images, and we ensured that it was able to generalize well to new data by using these techniques. With this approach, we achieved high accuracy in classifying bird species and demonstrated the effectiveness of transfer learning and other machine learning techniques for image recognition tasks.

## Model Performance
After training our model on the train set, we evaluated its performance on a validation set that was not used during training or testing. Our model achieved an accuracy of 95.8% in categorizing bird species, which is a high level of accuracy. Additionally, our model had a low loss of 0.163, indicating that it was able to generalize well to new data.

Overall, these results demonstrate that our transfer learning approach, combined with careful selection of hyperparameters and optimization techniques, was highly effective in training a model for bird species recognition. With high accuracy and low loss, our model is well-suited for practical applications, such as identifying bird species in the wild or in scientific research.

## Prediction
To predict the species of a bird, we have implemented two functions - "predict_pipeline_url" and "predict_bird_from_URL". These functions take the URL of an image as input, read the image from the URL, preprocess the image as necessary, and return the model's prediction of the bird species in the image.

The "predict_pipeline_url" function provides a complete pipeline for prediction, including loading the model, preprocessing the image, and returning the predicted bird species. This function is intended for use in applications when URL is passed as input and predicts species in the image as output, such as a web application that identifies bird species from images uploaded by users.

The "predict_bird_from_URL" function is a simpler function that is used when a user passes a URL to the function and the function prints the image and the species of the bird in the image.

## Conclusion
In this project, we have demonstrated how to categorize bird species images using transfer learning. We first pre-processed the images to ensure that they were suitable for input into the model. We then used the transfer learning technique to leverage the pre-existing knowledge of a pre-trained model, EfficientNetB0, to accurately classify the bird species images.

We trained the model using the Adam optimizer and categorical cross-entropy loss, and used techniques such as model checkpoint and early stopping to prevent overfitting. We achieved a high accuracy of 95.8% on the validation set, indicating that our model was effective in classifying bird species images.

Finally, we implemented two functions, "predict_pipeline_url" and "predict_bird_from_URL", to make it easy for users to predict the bird species from images using our trained model. These functions simplify the process of identifying bird species from images and provide accurate predictions with minimal effort.

Overall, this project demonstrates the power of transfer learning and deep learning techniques in accurately classifying bird species images, and provides a useful tool for researchers, bird enthusiasts, and anyone interested in identifying bird species from images.

## Deployment
I have used this model to incorporate it inside a basic HTML web page. Users can enter the URL of a bird image, and the web page will display the image from the URL along with the predicted bird species.

To create this functionality, I utilized HTML, CSS, JavaScript, and Python Flask. Flask is a web framework that allows us to create web applications using Python.

Below, you can see the website in action.

## Video and Notebook for reference:
Below is a video showcasing our model in action:

You can also find the notebook for this project:
1. By navigating to the [notebooks] folder in this repository
2. You can also find the notebook on [Kaggle](https://www.kaggle.com/code/jvkchaitanya410/bird-species-recognition-using-efficientnetb0), a popular platform for data science and machine learning.
