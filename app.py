import warnings
warnings.filterwarnings("ignore")
from flask import *  
import os
import cv2
import random
import numpy as np
from detect_book import *
import tensorflow as tf
import keras as k


app = Flask(__name__)  

session = tf.Session(graph = tf.Graph())
with session.graph.as_default():
    k.backend.set_session(session)
    m = k.models.load_model("weights.model")


@app.route("/")
def home():
    return render_template("file_upload_form.html",template_folder = "templates") 

@app.route('/success', methods = ['POST','GET'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        file = np.fromstring(f.read(),np.uint8)
        image = cv2.imdecode(file,cv2.IMREAD_COLOR)
        preprocessed_image = preprocessing(image)
        img = preprocessed_image[5:100,470:]
        blur = cv2.GaussianBlur(img,(1,1),0)
        ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.subtract(255,th)
        pred_img = th
        pred_img = cv2.resize(pred_img,(28,28))
        pred_img = pred_img/255.0
        pred_img = pred_img.reshape(1,28,28,1)
        mapping = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        with session.graph.as_default():
            k.backend.set_session(session)
            prediction = np.argmax(m.predict(pred_img),axis = 1)[0]
            name = mapping[prediction]
        if name == 'M':
            if not os.path.isdir("Meetings"):
                os.mkdir("Meetings")
                directory = "./Meetings/"
            else:
                directory = "./Meetings/"
        elif name == 'B':
            if not os.path.isdir("Bills"):
                os.mkdir("Bills")
                directory = "./Bills/"
            else:
                directory = "./Bills/"
        elif name == 'N':
            if not os.path.isdir("Notes"):
                os.mkdir("Notes")
                directory = "./Notes/"
            else:
                directory = "./Notes/"
        else:
            name = "Not a Given tag  ;-;"
        f.save(os.path.join(directory,f.filename))
        return render_template("success.html", name = f.filename)



if __name__ == '__main__':  
    app.run(host = '0.0.0.0',port = random.randint(1,9000), debug = True)  