#!/bin/bash
#
#

mkdir -p data/

curl -L -o dataset.zip\
  http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_big.zip

unzip dataset.zip -d data/
rm dataset.zip
