import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



app = Flask(__name__)
model = load_model('model_file.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(im):
    X = image.img_to_array(im)
    X = X/ 255
    X = np.expand_dims(X, axis = 0)
    return X

# Define a function to predict the image class
def predict_image_class(image_path):
    im = image.load_img(image_path, target_size=(150, 150))
    preprocessed_image = preprocess_image(im)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    return class_index

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join('static', 'uploaded_images', uploaded_file.filename)
            uploaded_file.save(file_path)
            class_index = predict_image_class(file_path)
            # Replace class_index with the actual class labels from your model
            class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'] 
            predicted_class = class_labels[class_index]
            return render_template('upload.html', filename=uploaded_file.filename, predicted_class=predicted_class)
    return render_template('upload.html', filename=None, predicted_class=None)

if __name__ == '__main__':
    app.run(debug=True)
