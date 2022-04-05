#!/bin/bash

wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip" -O CMU_MOSI.zip
unzip CMU_MOSI.zip
rm CMU_MOSI.zip
mv Raw/ CMU_MOSI

wget "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd" -O CMU_MOSI/CMU_MOSI_Opinion_Labels.csd