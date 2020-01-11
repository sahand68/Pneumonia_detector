FROM python:3.6.10-stretch

COPY . /app
WORKDIR /app

RUN pip install Werkzeug scikit-learn matplotlib pydicom Flask==1.0.2 numpy gevent pillow h5py tensorflow


EXPOSE 5000
CMD [ "python" , "predict_cpu.py"]
