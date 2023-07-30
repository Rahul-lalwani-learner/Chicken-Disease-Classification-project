from flask import Flask, render_template, request, redirect, jsonify
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictPipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

class ClientApp: 
    def __init__(self): 
        self.filename = "inputImage.jpg"
        self.classifier = PredictPipeline(self.filename)

    
@app.route("/", methods = ["GET"])
@cross_origin()
def home(): 
    return render_template("index.html")

@app.route("/train", methods = ["POST", "GET"])
@cross_origin()
def trainRoute(): 
    os.system("dvc repro")
    return "Training done Successfully"


@app.route("/predict", methods = ["POST", "GET"])
@cross_origin()
def predictRoute():
    image = request.json["image"]
    decodeImage(image, App.filename)
    result = App.classifier.predict()
    return jsonify(result)

if __name__ == "__main__": 
    App = ClientApp()
    app.run(host="0.0.0.0", port=5000, debug=True)