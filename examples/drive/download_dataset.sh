#!/bin/bash
#
#

mkdir -p data/

curl -L -o drive.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/drive-digital-retinal-images-for-vessel-extraction

unzip drive.zip -d data/
