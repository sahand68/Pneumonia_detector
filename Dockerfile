FROM python:3.6.10-stretch

COPY . /opt/app
WORKDIR /opt/app

RUN pip install -r requirements.txt


EXPOSE 5000
CMD [ "python" , "app.py"]
