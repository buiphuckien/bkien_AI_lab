# import pandas as pd
import numpy as np
# import sklearn
import joblib
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/') # to homepage
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST': # If there is post
        a = float(request.form['a'])
        b = float(request.form['b'])
        test = np.array([a, b]) # shape = (2,)
        test = test.reshape(1, -1) # shape = (1, 2)
        model = open('lr_model.pkl', 'rb') # open the model
        lr_model = joblib.load(model) # load model
        pred  = lr_model.predict(test)
        return render_template('predict.html', pred=pred)
    else:
        return render_template('predict.html', pred='None')
    
if __name__=='__main__':
    app.run(debug=True)


        
