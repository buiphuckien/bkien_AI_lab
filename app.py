import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import sklearn
import joblib
from PIL import Image
from dnn_app_utils_v3 import predict, L_model_forward
app = Flask(__name__)

@app.route('/') # to index page
def home():
    return render_template('index.html')

@app.route('/about') # to about
def about():
    return render_template('about.html')

@app.route('/contact') # to contact
def contact():
    return render_template('contact.html')

@app.route('/post') # to post
def post():
    return render_template('post.html')

@app.route('/math') # to math
def math():
    return render_template('math.html')

@app.route('/predict_lr', methods=['GET', 'POST'])
def predict_lr():
    # print(request)
    if request.method == 'POST': # If there is post

        # Get values from form
        a = float(request.form['a']) 
        b = float(request.form['b'])

        # Putting a, b into an array
        test = np.array([a, b]) # shape = (2,)
        test = test.reshape(1, -1) # shape = (1, 2)

        # Loading the model
        model = open('lr_model.pkl', 'rb') # open the model
        lr_model = joblib.load(model) # load model

        # Make a prediction
        pred  = lr_model.predict(test)
        pred = pred.item() # Just

        # Give answer to a prediction
        return render_template('predict_lr.html', pred=pred)
    
    else:
        return render_template('predict_lr.html')




UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print(request.files)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            image = np.array(Image.open(file).resize((64, 64)))

            image = image / 255.
            image = image.reshape((1, 64 * 64 * 3)).T
            # print(image)
            my_label_y = [1]



            with open('parameters.pkl', 'rb') as f:
                parameters = joblib.load(f) # Load the dictionary for reuse


            p, _ = L_model_forward(image, parameters)
            print('p ===== ', p)


            p = round(np.squeeze(p)*100, 2)

            
            
            # my_predicted_image = predict(image, my_label_y, parameters)


            # print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + str(int(np.squeeze(my_predicted_image))) +  "\" picture.")



            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            

            return render_template('predict.html', pred=p)

            # return redirect(url_for('upload_file',
            #                         filename=filename))
    return '''
    <!doctype html>
    <title>cat vs noncat</title>
    <h1>Upload an image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''