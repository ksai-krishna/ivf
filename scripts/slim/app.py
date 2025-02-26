from flask import Flask, render_template, request
import os
import pred # Import predict.py as module

app = Flask(__name__)

# --- Configuration ---
# Modify these to match your setup
TRAINED_MODEL_DIR = 'result' # Replace with your trained model directory # Changed to 'result' to match user example
ARCHITECTURE = 'v1' # or 'v3' or 'inception_resnet2' - must match training and predict.py # Changed to 'v1' to match user example
UPLOAD_FOLDER = 'uploads' # Folder to save uploaded images temporarily
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload folder exists


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_route():
    if 'image' not in request.files:
        return render_template('index.html', message='No image part')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', message='No selected image')
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename) # Save the uploaded image temporarily

        try:
            print("Before calling predict_image_label in route") # ADDED PRINT for debugging
            prediction = pred.predict_image_label(TRAINED_MODEL_DIR, "../../Images/test/good_23765483_-15_3AA.jpg", ARCHITECTURE) # Call function from predict.py
            print("After calling predict_image_label in route, prediction:", prediction) # ADDED PRINT for debugging
            return render_template('result.html', prediction=prediction, image_path=filename)

        except Exception as e:
            print(f"Exception in predict_image_route: {e}") # ADDED PRINT for debugging
            return render_template('result.html', error=str(e), image_path=filename)
    else:
        return render_template('index.html', message='Allowed image types are png, jpg, jpeg')


if __name__ == '__main__':
    print("Starting Flask app.run()") # ADDED PRINT for debugging
    app.run(debug=True)