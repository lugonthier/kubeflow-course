FROM python:3.11-slim


WORKDIR /usr/src/app


RUN pip install --no-cache-dir tensorflow==2.15.0

COPY train.py .

CMD ["python", "./train.py"]