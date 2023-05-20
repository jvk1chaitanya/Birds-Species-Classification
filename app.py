from flask import Flask,request,render_template,jsonify

from src.pipeline.predict_pipeline import predict_species
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def home():
    try:
        logging.info("Rendering the home page")
        return render_template('index.html')

    except Exception as e:
        logging.info("Error in rendering the home page")
        raise CustomException(e)

@app.route('/classify',methods=['GET','POST'])
def classify():
    try:
        if request.method == 'GET':
            logging.info("Rendering the classify page")
            return render_template('species-predict.html')
        
        if request.method == 'POST':
            logging.info("Starting the classification")

            # get the image from the request
            image = request.json['url']

            # get the predicted class
            predicted_class = predict_species.predict_pipeline_url(image)

            #logging.info("Classification completed successfully")
            return jsonify({'class': predicted_class})

    except Exception as e:
        # Return the error message as JSON
        logging.info("Error in classifying the image")
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)