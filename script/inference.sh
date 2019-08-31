#!/bin/bash
# FALCON: FAst and Lightweight CONvolution
#
# Authors:
#  - Chun Quan (quanchun@snu.ac.kr)
#  - U Kang (ukang@snu.ac.kr)
#  - Data Mining Lab. at Seoul National University.
#
# File: scripts/demo.sh
#  - Test trained model
#  - Trained model saved in ../train_test/trained_model.
#
# Version: 1.0
#==========================================================================================
cd ../src/train_test

CUDA_VISIBLE_DEVICES=0 python main.py -m VGG19 -conv FALCON -init -data cifar10;
