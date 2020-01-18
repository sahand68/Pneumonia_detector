

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import pydicom
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import measure
from skimage.transform import resize
import csv
import random
# from flask import Flask, redirect, url_for, request, render_template, send_file, jsonify
from gevent.pywsgi import WSGIServer
import cv2
from werkzeug.utils import secure_filename

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader(['./templates']))

from sanic import Sanic, response
app = Sanic(__name__)

app.static('/static', './static')

# Any results you write to the current directory are saved as output.

# Define a flask app

@app.route('/', methods=['GET'])
def index(request):
    data = {'name': 'name'}
    template = env.get_template('index.html')
    html_content = template.render(name=data["name"])
    # Main page
    return response.html(html_content)



# In[4]:


def parse_data(df, test = False):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    if not test:
      for n, row in df.iterrows():
          # --- Initialize patient entry into parsed 
          pid = row['patientId']
          if pid not in parsed:
              parsed[pid] = {
                  'dicom': 'data/stage_2_train_images/%s.dcm' % pid,
                  'label': row['Target'],
                  'boxes': []}

          # --- Add box if opacity is present
          if parsed[pid]['label'] == 1:
              parsed[pid]['boxes'].append(extract_box(row))
    else:
      for n, row in df.iterrows():
          # --- Initialize patient entry into parsed 
          pid = row['patientId']
          if pid not in parsed:
              parsed[pid] = {
                  'dicom': 'uploads/%s.dcm' % pid,
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
        #rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = [255, 251, 204] # Just use yellow
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=15)

    file_name = '{}_detected.png'.format(data['dicom'].split('.')[0])
    cv2.imwrite(file_name, im)
    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')
    return file_name

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


# In[5]:


import tensorflow as tf
class generator(tf.keras.utils.Sequence):
    
    def __init__(self, folder, filenames, nodule_locations=None, batch_size=32, image_size=512, shuffle=True, predict=False, augment = False):
        self.folder = folder
        self.filenames = filenames
        self.nodule_locations = nodule_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        nodule_locations={}
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains nodules
        if filename in nodule_locations:
            # loop through nodules
            for location in nodule_locations[filename]:
                # add 1's at the location of the nodule
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)


# In[6]:


def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
def create_downsample(channels, inputs):
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LeakyReLU(0)(x)
    x = tf.keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LeakyReLU(0)(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0)(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return tf.keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=5):
    # input
    inputs = tf.keras.Input(shape=(input_size, input_size, 1))
    x = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0)(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = tf.keras.layers.UpSampling2D(2**depth)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# In[9]:


def predict(request):
    f = request.files.get('file')

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.name))
    write = open(file_path, 'wb')
    write.write(f.body)
    k_=[]
    x_= []
    y_ =[]
    w_ =[]
    h_ =[]
    t_= []
    area = []
    # create test generator with predict flag set to True
    test_gen = generator('uploads' ,[f.name], None, batch_size=1, image_size=512, shuffle=False, predict=True)
    for imgs, filenames in test_gen:
        # predict batch of images
        model = create_network(input_size=512, channels=32, n_blocks=2, depth=4)
        model.load_weights("model/model.h5")
        preds = model.predict(imgs)
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (1024, 1024), mode='reflect')
            # threshold predicted mask
            comp = pred[:, :, 0] > 0.3
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                k_.append(filename.split('.')[0])
                x_.append(x)
                y_.append(y)
                h_.append(height)
                width = x2 - x
                w_.append(width)
                conf = np.mean(pred[y:y+height, x:x+width])
                area.append(width*height)
                t_.append(conf)
        # if len(x_) >= len(test_filenames):
        #     break
    test_predictions = pd.DataFrame()
    test_predictions['patientId'] = k_
    test_predictions['x'] =x_
    test_predictions['y'] =y_
    test_predictions['width'] =w_
    test_predictions['height']=h_
    test_predictions['Target'] = t_
    test_predictions['area'] = area
    return test_predictions


# ### testing shows that model is not trained enough to make predictions
# 

# In[ ]:

@app.route('/predict', methods=['GET', 'POST'])
def make_preds(request):
    test_predictions = predict(request)
    status = 'detected.'
    if test_predictions['Target'].any():
        test_predictions['Target'].values[test_predictions['Target'].values > 0.3] = 1        
        print('Pneumonia positive')
        parsed_test= parse_data(test_predictions, test = True)
        plt.style.use('default')
        fig=plt.figure(figsize=(12, 20))
        file_name=draw(parsed_test[test_predictions['patientId'].unique()[0]])
    else:
        status = 'not detected.'
        f = request.files.get('file')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.name))
        d = pydicom.read_file(file_path)
        im = d.pixel_array
        file_name = "uploads/{}.png".format(secure_filename(f.name).split('.')[0])
        cv2.imwrite(file_name, im)
        plt.imshow(im, cmap=plt.cm.gist_gray)
    return response.json({
        'file_name': file_name,
        'status': status,
        # 'confidence': confidence
    })



# Callback to grab an image given a local path
@app.route('/get_image')
def get_image(request):
    path = request.args.get('p')
    _, ext = os.path.splitext(path)
    exists = os.path.isfile(path)
    if exists:
        return response.file(path, mime_type='image/' + ext[1:])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, access_log=False, workers=1)

# In[ ]:




