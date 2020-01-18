FROM python:3.7
COPY . /app
COPY requirements.txt /opt/app/requirements.txt

WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000
CMD [ "python" , "app.py"]
