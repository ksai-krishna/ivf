import tensorflow as tf

import tf_slim as slim
#slim = tf.contrib.slim
import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing
from os import listdir
from os.path import isfile, join
from os import walk
os.environ['CUDA_VISIBLE_DEVICES'] = '' #Uncomment this line to run prediction on CPU.
session = tf.compat.v1.Session()
deep_lerning_architecture = None # to be set dynamically
nb_classes = 2 # Fixed to 2 classes (good/poor)


tf.compat.v1.disable_eager_execution()  # Disable eager execution for TF 1.x
graph = tf.Graph()
session = tf.compat.v1.Session(graph=graph)


def get_test_images(image_path): # Modified to take single image path
    return [image_path] # Return the single image path in a list

def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
        image_size = get_image_size_for_architecture(deep_lerning_architecture) # Get image size based on architecture
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]

def get_image_size_for_architecture(architecture):
    if architecture == "v1" or architecture == "V1":
        return 224
    elif architecture == "v3" or architecture == "V3" or architecture == "resv2" or architecture == "inception_resnet2":
        return 299
    else:
        raise ValueError("Invalid architecture. Choose from v1, v3, or inception_resnet2")

def predict_image_label(model_dir, image_path, architecture):
    global deep_lerning_architecture
    deep_lerning_architecture = architecture

    with graph.as_default():  # Ensure we use the correct TensorFlow graph
        with session.as_default():
            image_size = get_image_size_for_architecture(deep_lerning_architecture)

            processed_images = tf.compat.v1.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

            if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
                with slim.arg_scope(inception.inception_v1_arg_scope()):
                    logits, _ = inception.inception_v1(processed_images, num_classes=nb_classes, is_training=False)
            else:
                if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3":
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits, _ = inception.inception_v3(processed_images, num_classes=nb_classes, is_training=False)
                else:
                    if deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
                        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                            logits, _ = inception.inception_resnet_v2(processed_images, num_classes=nb_classes, is_training=False)

            probabilities = tf.nn.softmax(logits)

            checkpoint_path = tf.train.latest_checkpoint(model_dir)
            init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())
            init_fn(session)

            print('Start to transform image!')
            image_list = get_test_images(image_path)
            images = transform_img_fn(image_list)

            print('Start doing prediction!')
            preds = session.run(probabilities, feed_dict={processed_images: images})

            predicted_class_index = np.argmax(preds[0, :])
            classes = ['good', 'poor']
            predicted_class_label = classes[predicted_class_index]

            print("Prediction completed:", predicted_class_label)
            return predicted_class_label


if __name__ == '__main__':

    if len(sys.argv) != 5: # Modified to expect 5 arguments now (including script name)
        print("The script needs four arguments.") # Updated message
        print("The first argument should be the CNN architecture: v1, v3 or inception_resnet2")
        print("The second argument should be the directory of trained model.")
        print("The third argument should be the path to the input image.") # Changed argument description
        print("The fourth argument should be output file for predictions.") # Keep output file argument - although not used now
        #print("The fifth argument should be number of classes.") # No longer needed as fixed to 2 classes (good/poor)
        exit()
    deep_lerning_architecture_cli = sys.argv[1] # architecture from command line
    train_dir_cli = sys.argv[2] # model dir from command line
    image_path_cli = sys.argv[3] # image path from command line
    output_file_cli = sys.argv[4] # output file from command line - not used in logic but expected as argument

    predicted_label = predict_image_label(train_dir_cli, image_path_cli, deep_lerning_architecture_cli) # Call the function
    print(predicted_label) # print for command line