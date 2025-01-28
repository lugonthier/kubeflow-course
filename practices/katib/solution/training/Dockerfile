FROM python:3.10-slim


WORKDIR /usr/src/app


RUN pip install --no-cache-dir tensorflow==2.13.0 tensorflow-datasets


COPY train.py .

CMD ["python", "./train.py"]