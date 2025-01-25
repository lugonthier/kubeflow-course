FROM tensorflow/tensorflow:2.15.0

RUN pip install tensorflow-datasets google-cloud-storage

COPY train.py .



CMD ["python", "./train.py"]