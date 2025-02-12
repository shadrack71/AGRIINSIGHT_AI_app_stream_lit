from flask import Flask, request, make_response, render_template, redirect, url_for, Response, send_from_directory, \
    jsonify, session, flash
import logic_controller

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

logic_controller = logic_controller.Data_controller()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/agriinsight_ai')
def agriinsight():
    crop_dir = 'image/crop (30).JPG'
    crop_result = logic_controller.model_prediction(crop_dir)
    crop_recommendation = logic_controller.recommendation_management(crop_result)
    return render_template('agriinsight.html',crop_result=crop_result,crop_recommendation=crop_recommendation)

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)