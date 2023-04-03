from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)
# import ridge regressor model and  standar scalar pickel file
ridge_model = pickle.load(open("Models/ridge.pkl", "rb"))
StandardScalar_s = pickle.load(open("Models/StandardScalar.pkl", "rb"))

# Route for home page


# @app.route("/")
# def home_page():
#     return render_template("index.html")


@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = request.form.get('Temperature')
        RH = request.form.get('RH')
        Ws = request.form.get('Ws')
        Rain = request.form.get('Rain')
        FFMC = request.form.get('FFMC')
        DMC = request.form.get('DMC')
        ISI = request.form.get('ISI')
        Classes = request.form.get('Classes')
        Region = request.form.get('Region')

        new_data_sc = StandardScalar_s.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_sc)

        return render_template("home.html", result=result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
