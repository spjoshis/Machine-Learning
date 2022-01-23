FROM ubuntu:latest

RUN apt-get -y update
RUN apt-get install python3 -y
RUN apt-get install pip -y

COPY ./*.ipynb /application/
COPY ./requirements.txt /application/

WORKDIR /application/
RUN pip install -r requirements.txt

EXPOSE 8888
