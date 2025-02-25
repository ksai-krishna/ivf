from flask import Flask, render_template, request
import tensorflow as tf
import tf_slim as slim
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing

app = Flask(__name__)

# --- Configuration ---
# Modify these to match your setup
TRAINED_MODEL_DIR = '../scripts/slim/result' # Replace with your trained model directory
ARCHITECTURE = 'v3' # or 'v1' or 'inception_resnet2' - as used for training
UPLOAD_FOLDER = 'uploads' # Folder to save uploaded images temporarily
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload folder exists

# TensorFlow Session (Initialize outside the function to reuse across requests)
session = tf.compat.v1.Session()
deep_lerning_architecture = ARCHITECTURE # Set architecture from config for prediction functions
nb_classes = 2 # Fixed classes
image_size = 0 # Will be set based on architecture

# Get image size based on architecture (moved from predict.py)
def get_image_size_for_architecture(architecture):
    if architecture == "v1" or architecture == "V1":
        return 224
    elif architecture == "v3" or architecture == "V3" or architecture == "resv2" or architecture == "inception_resnet2":
        return 299
    else:
        raise ValueError("Invalid architecture. Choose from v1, v3, or inception_resnet2")

image_size = get_image_size_for_architecture(deep_lerning_architecture) # Set global image_size

# Image transformation function (moved from predict.py)
def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]

# Prediction function (integrated from predict.py)
def predict_image_class(image_path):
    image_list = [image_path] # Create list for transform function
    tf.compat.v1.disable_eager_execution()
    processed_images = tf.compat.v1.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

    if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(processed_images, num_classes=nb_classes, is_training=False)
    elif deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3":
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, _ = inception.inception_v3(processed_images, num_classes=nb_classes, is_training=False)
    elif deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            logits, _ = inception.inception_resnet_v2(processed_images, num_classes=nb_classes, is_training=False)
    else:
        raise ValueError("Invalid architecture in configuration.")


    probabilities = tf.nn.softmax(logits)
    checkpoint_path = tf.train.latest_checkpoint(TRAINED_MODEL_DIR)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,slim.get_variables_to_restore())
    init_fn(session) # Initialize session only once when app starts

    images = transform_img_fn(image_list)
    preds = session.run(probabilities, feed_dict={processed_images: images})

    predicted_class_index = np.argmax(preds[0,:])
    classes = ['good', 'poor']
    predicted_class_label = classes[predicted_class_index]
    return predicted_class_label

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_route(): # Renamed to avoid conflict with predict_image_class function
    if 'image' not in request.files:
        return render_template('index.html', message='No image part')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', message='No selected image')
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename) # Save the uploaded image temporarily

        try:
            prediction = predict_image_class(filename) # Call prediction function directly
            return render_template('result.html', prediction=prediction, image_path=filename)

        except Exception as e:
            return render_template('result.html', error=str(e), image_path=filename)
    else:
        return render_template('index.html', message='Allowed image types are png, jpg, jpeg')

# Initialize TensorFlow graph and session when app starts (outside request context)
tf.compat.v1.disable_eager_execution()
processed_images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_for_init, _ = inception.inception_v1(processed_images_placeholder, num_classes=nb_classes, is_training=False)
elif deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3":
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits_for_init, _ = inception.inception_v3(processed_images_placeholder, num_classes=nb_classes, is_training=False)
elif deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits_for_init, _ = inception.inception_resnet_v2(processed_images_placeholder, num_classes=nb_classes, is_training=False)

probabilities_op = tf.nn.softmax(logits_for_init) # Define probabilities op for session.run

checkpoint_path_init = tf.train.latest_checkpoint(TRAINED_MODEL_DIR)
init_fn_init = slim.assign_from_checkpoint_fn(checkpoint_path_init,slim.get_variables_to_restore())
init_fn_init(session) # Initialize session with checkpoint weights only once

if __name__ == '__main__':
    app.run(debug=True)