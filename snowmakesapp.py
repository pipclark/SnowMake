# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:04:54 2021

@author: Pip
"""

from flask import Flask, request, json, render_template, make_response, send_file
import base64
from flask_cors import CORS
from snowflakebuilder import conditions, flakegrower, flake_video

#%%

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/advanced')
def index():
    return render_template('advanced.html')

@app.route('/howitworks')
def index():
    return render_template('howitworks.html')

@app.route('/about')
def index():
    return render_template('about.html')


@app.route('/randomflake')
def random_flake_api():
    print(f'starting growth at {datetime.now()}')
    growthcons = conditions()
            
    flake,corner1,centre1 = flakegrower(growthcons)
            
    animationfile = flake_video(flake,corner1,centre1)
    
    print('flake made, about to send')

    with open(animationfile, 'rb') as f:
        image_binary = f.read()
    print('binary read in, about to delete file and send binary')
    os.remove(animationfile) # delete file after it's been read
        
    response = make_response(base64.b64encode(image_binary))
    response.headers.set('Content-Type', 'image/gif')
    response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
    return response


if __name__ == "__main__":
    app.run()