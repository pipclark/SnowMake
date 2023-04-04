# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:04:54 2021

@author: Pip
"""

from flask import Flask, request, json, render_template, make_response, send_file
import base64
from flask_cors import CORS
from snowflakebuilder import generate_path_conditions, grow_snowflakes_along_path, generate_snowflake_gif, generate_conditions_video, generate_conditions_graph_lines
from datetime import datetime
import os
import ast


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/advanced')
def advanced_mode():
    return render_template('advanced.html')


@app.route('/howitworks')
def how_it_works():
    return render_template('howitworks.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/randomflake', methods=['GET', 'POST'])
def random_flake_api():
    print(f'starting growth at {datetime.now()}')
    if request.method == 'POST':
        # convert the post data imported in a binary format to a dictionary
        # (part in brackets converts to a string, ast_lit.. part converts string rep of dictionary to dictionary
        options = ast.literal_eval(request.data.decode(
            'UTF-8'))
        print(options)

        growth_conditions, T, over100RHu, startidx, h = generate_path_conditions(options)

        flake, corner1, centre1 = grow_snowflakes_along_path(growth_conditions, options)
        animation_file = generate_snowflake_gif(flake, corner1, centre1, options)

    else:
        # randomly generate growth conditions
        growth_conditions = generate_path_conditions()
        # grow the snowflake - find
        flake, corner1, centre1 = grow_snowflakes_along_path(growth_conditions)
        # turn the growth
        animation_file = generate_snowflake_gif(flake, corner1, centre1)

    print('flake made, about to send')

    with open(animation_file, 'rb') as f:
        image_binary = f.read()
    print('binary read in, about to delete file and send binary')

    os.remove(animation_file)  # delete file after it's been read

    # send back the snowflake gif
    response = make_response(base64.b64encode(image_binary))
    response.headers.set('Content-Type', 'image/gif')
    response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
    return response


@app.route('/advflakegraphs', methods=['GET', 'POST'])
def adv_graphs_api():
    if request.method == 'POST':
        options = ast.literal_eval(request.data.decode(
            'UTF-8'))
        print(options)

        growthcons, T, over100RHu, startidx, h = generate_path_conditions(options)
        extra_graphs_file = generate_conditions_video(h, T, over100RHu, startidx)
    else:
        print('no advanced inputs recieved')

    print('flake made, about to send')
    with open(extra_graphs_file, 'rb') as f:
        image_binary = f.read()
    print('binary read in, about to delete file and send binary')
    os.remove(extra_graphs_file)  # delete file after it's been read

    response = make_response(base64.b64encode(image_binary))
    response.headers.set('Content-Type', 'image/gif')
    response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
    return response


if __name__ == "__main__":
    app.run()
