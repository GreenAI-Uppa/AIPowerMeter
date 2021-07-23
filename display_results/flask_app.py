#https://testdriven.io/blog/developing-a-single-page-app-with-flask-and-vuejs/
import datetime
from display_results import driver
from flask import Flask, render_template, jsonify
import json, pandas as pd
from flask_cors import CORS

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
    )

app = Flask(__name__)


driver = driver.Driver()

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route("/")
def index():
    return render_template("display_curves.html")

@app.route("/get_models", methods=['GET'])
def get_model():
    print("recieved call on models")
    return driver.get_models()


@app.route("/get_curves", methods=['GET'])
def get_curves():
    print("recieved call")
    power_nvidia = driver.get_curve('nvidia_draw_absolute')
    accuracy = driver.get_curve('test_accuracy')
    power_nvidia, accuracy = driver.interpolate(power_nvidia, accuracy)
    return {"power_nvidia":power_nvidia, "accuracy":accuracy}

if __name__ == '__main__':
    app.run(host = 'localhost', port=5001)
