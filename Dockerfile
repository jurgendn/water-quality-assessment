FROM python:3.11.7-slim-bookworm

RUN apt-get update
RUN apt-get install -y cmake build-essential

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
