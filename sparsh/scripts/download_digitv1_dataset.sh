#!/bin/bash

# set path to the directory to donwload the dataset
path_dataset="/media/chiguera/GUM/datasets/"

mkdir -p $path_dataset/digitv1
cd $path_dataset/digitv1

echo "Downloading the backbone dataset for DIGITv1..."
gdown https://drive.google.com/drive/folders/19vs-5dSqakiJ96ykBdHbhDuc8EoYK0eg?usp=sharing --folder

# extract files
cd ./Object-Slide
for i in */; do tar -xvf "${i%/}.tar.gz"; done
rm *.tar.gz

echo "Done!"
