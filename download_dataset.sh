#!/bin/bash
mkdir -p ./data

curl -L -o ./data/flood-area-segmentation.zip \
  https://www.kaggle.com/api/v1/datasets/download/faizalkarim/flood-area-segmentation

unzip ./data/flood-area-segmentation.zip -d ./data/

rm ./data/flood-area-segmentation.zip

echo "Download and extraction complete!"
