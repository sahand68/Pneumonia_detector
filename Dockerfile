FROM python:3.6.10-stretch

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install Werkzeug scikit-learn matplotlib pydicom Flask numpy gevent pillow h5py tensorflow


EXPOSE 5000
CMD [ "python" , "predict_cpu.py"]
