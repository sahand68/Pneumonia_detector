import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pydicom
import os
from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template, send_file

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
verbose = True

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/model.h5'
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(images_path,model):
    # images_path = 'data/stage_2_test_images'
    predictions = pd.DataFrame()


    img = pydicom.dcmread(images_path).pixel_array
    # resize image
    img = resize(img, (256, 256), mode='reflect')
    # add trailing channel dimension
    img = np.expand_dims(img, -1)
    # test_filenames  = os.listdir(images_path)
    test_gen = generator(images_path, test_filenames, None, batch_size=25, image_size=256, shuffle=False, predict=True)
    with tf.device("/gpu:0"):
        preds = model.predict(img)
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        comp = pred[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        for region in measure.regionprops(comp):
                    # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            predictions['patientId'] = filename.split('.')[0])
            predictions['x']=x
            predictions['y'] = y
            predictions['height'] = height
            predictions['width']  = x2 - x
            w_.append(width)
            conf = np.mean(pred[y:y + height, x:x + width])
            predictions['Target'] = conf
            predictions['Target'].values[predictions['Target'].values > 0.5] = 1

    return predictions

def parse_data(df):
    parsed = {}
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': 'data/stage_2_test_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


def draw(data):
    """
    Method to draw single patient with bounding box(es) if present

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = [255, 251, 204]  # Just use yellow
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=15)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')


def overlay_box(im, box, rgb, stroke=2):
    """
    Method to overlay single box on image
    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]

    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Output file path
        file_output_path = 'os.path.join(basepath, 'uploads', 'predictions.csv')'
        # Make prediction
        preds = model_predict(file_path, model)
        parsed_test = parse_data(preds)
        # Write to uploads directory
        plt.style.use('default')
        fig = plt.figure(figsize=(12, 20))
        draw(parsed_test)
        plt.show()
        preds.to_csv(file_output_path, index=False)
        # Delete uploaded file
        os.remove(file_path)
        return file_output_path
    return None

# Callback to grab an image given a local path
@app.route('/get_image')
def get_image():
    path = request.args.get('p')
    _, ext = os.path.splitext(path)
    exists = os.path.isfile(path)
    if exists:
        return send_file(path, mimetype='image/' + ext[1:])


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()




