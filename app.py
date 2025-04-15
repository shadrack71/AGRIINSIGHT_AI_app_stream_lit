from flask import Flask, request, make_response, render_template, redirect, url_for, Response, send_from_directory, \
    jsonify, session, flash
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import logic_controller

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

logic_controller = logic_controller.Data_controller()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/agriinsight_ai')
def agriinsight():
    # Get the file path from the query parameter
    crop_dir = request.args.get('filepath')
    if not crop_dir:
        return redirect(url_for('upload_file'))

   # crop_dir = 'image/crop (30).JPG'
    crop_result = logic_controller.model_prediction(crop_dir)
    crop_recommendation = logic_controller.recommendation_management(crop_result)
    filter_plant_name = logic_controller.plant_name_mapping(crop_result)
    return render_template('agriinsight.html',crop_result=crop_result,crop_recommendation=crop_recommendation, filter_plant_name = filter_plant_name,uploaded_image=crop_dir)


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser submits empty file
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save file
            file.save(filepath)

            # Redirect to agriinsight route with the file path as parameter
            return redirect(url_for('agriinsight', filepath=filepath))

    return render_template('upload.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)