FROM tensorflow/tensorflow:latest-gpu

WORKDIR /player-pricing

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip

COPY requirements.txt /player-pricing/
RUN pip3 install -r requirements.txt

COPY . /player-pricing/
