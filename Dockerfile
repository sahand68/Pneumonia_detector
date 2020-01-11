FROM python:3.6.10-stretch

COPY . /app
WORKDIR /app

RUN pip install Werkzeug scikit-learn matplotlib pydicom Flask numpy gevent pillow h5py tensorflow opencv-contrib-python


EXPOSE 5000
CMD [ "python" , "app.py"]
