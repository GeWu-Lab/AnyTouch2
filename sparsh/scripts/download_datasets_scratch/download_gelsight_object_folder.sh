#!/bin/bash

# set path to the directory to donwload the dataset
path_dataset="/media/chiguera/GUM/datasets/gelsight/object_folder/"

echo "Downloading the backbone dataset for gelsight (no markers)..."
cd $path_dataset

# replace X with a value in [10,20,30,40,50,60,70,80,90]

wget https://download.cs.stanford.edu/viscam/ObjectFolder_Real/tactile/tactile_data_11_20.tar.gz
tar -xvf tactile_data_1_10.tar.gz