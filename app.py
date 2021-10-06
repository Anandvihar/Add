# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:28:15 2021

@author: Anandvihar
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
model4 = pickle.load(open('model4.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index1.html')

@app.route("/predict", methods=["POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input1 = [int(x) for x in request.form.values()]
    Addition1 = [np.array(input1)]
    result = model4.predict(Addition1)

    output = round(result[-1], 2)

    return render_template('index1.html', result_check ='Addition of a and b is {}'.format(output))


if __name__ == "__main__":
    #app_1.run(host='150.129.130.254',port='8000')
    #app_1.run(host='0.0.0.1', port='8888')
    app.run()
    #debug=True