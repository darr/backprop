#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2018-12-23 16:53
# Modified date : 2018-12-28 14:05
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import mnist_dataset
import forward_neural_network

def test_mnist_set():
    file_path = "./data/"
    batch_size = 8
    dataset = mnist_dataset.MnistSet(file_path)
    data_generator = dataset.get_train_data_generator(batch_size)
    batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
    print("status:%s" % status)
    print(str(batch_labels))
    print("batch_img size:%s" % len(batch_img))

def test_deep_model_with_epochs():
    neural_model = forward_neural_network.DeepModel()
    neural_model.show_hyperparameters()
    neural_model.run_with_epoch()

def test_deep_model_with_steps():
    neural_model = forward_neural_network.DeepModel()
    neural_model.show_hyperparameters()
    neural_model.run_with_steps()

def run():
    #test_mnist_set()
    #test_deep_model_with_epochs()
    test_deep_model_with_steps()

run()

