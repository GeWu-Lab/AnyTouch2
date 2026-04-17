#!/bin/bash

# set path to the directory to donwload the dataset
path_dataset="/media/chiguera/GUM/datasets/"

mkdir -p $path_dataset/gelsight
cd $path_dataset/gelsight

# download object_folder dataset
echo "Downloading the backbone dataset gelsight/object_folder"
gdown https://drive.google.com/drive/folders/1kgKj3BhvSN8bF1hI2bjeqhcaCHyxJUss?usp=sharing --folder


# download touch_go dataset
echo "Downloading the backbone dataset gelsight/touch_go"
gdown https://drive.google.com/drive/folders/1Rpy9ZHCfJjwycj7TMuEbwwHEtH79ls8D?usp=sharing --folder

# extract files
cd ./touch_go
for i in */; do tar -xvf "${i%/}.tar.gz"; done
rm *.tar.gz

echo "Done!"
