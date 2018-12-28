#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : mnist_dataset.py
# Create date : 2018-12-24 19:58
# Modified date : 2018-12-28 13:36
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

#http://yann.lecun.com/exdb/mnist/

import sys
import struct
import numpy as np

class MnistSet(object):
    def __init__(self, file_path):
        super(MnistSet, self).__init__()
        # pylint: disable=bad-continuation
        self._file_list = [
                            "train-images-idx3-ubyte",
                            "train-labels-idx1-ubyte",
                            "t10k-images-idx3-ubyte",
                            "t10k-labels-idx1-ubyte",
                            ]
        # pylint: enable=bad-continuation
        self.file_path = file_path

    @property
    def file_list(self):
        return self._file_list

    def open_file_with_full_name(self, full_path, open_type):
        try:
            file_object = open(full_path, open_type)
            return file_object
        except Exception as e:
            print(e)
            return None

    def get_file_full_name(self, path, name):
        if path[-1] == "/":
            full_name = path +  name
        else:
            full_name = path + "/" +  name
        return full_name

    def open_file(self, path, name, open_type='a'):
        file_name = self.get_file_full_name(path, name)
        return self.open_file_with_full_name(file_name, open_type)

    def _read_mnist(self, file_name):
        file_object = self.open_file(self.file_path, file_name, open_type="rb")
        return file_object

    def _get_file_header_data(self, file_obj, header_len, unpack_str):
        raw_header = file_obj.read(header_len)
        header_data = struct.unpack(unpack_str, raw_header)
        return header_data

    def _read_a_image(self, file_object):
        raw_img = file_object.read(28*28)
        img = struct.unpack(">784B", raw_img)
        return img

    def _read_a_label(self, file_object):
        raw_label = file_object.read(1)
        label = struct.unpack(">B", raw_label)
        return label

    def _generate_a_batch(self, images_file_name, labels_file_name, batch_size):
        images_file = self._read_mnist(images_file_name)
        header_data = self._get_file_header_data(images_file, 16, ">4I")
        labels_file = self._read_mnist(labels_file_name)
        header_data = self._get_file_header_data(labels_file, 8, ">2I")

        ret = True
        while True:
            images = np.zeros(shape=(batch_size, 784))
            labels = np.zeros(shape=(batch_size, 10))
            for i in range(batch_size):
                try:
                    image = self._read_a_image(images_file)
                    label = self._read_a_label(labels_file)
                    images[i] = image
                    labels[i][label] = 1
                except Exception as err:
                    #print(err)
                    ret = False
                    break
            yield images, labels.astype(int), ret

    def get_train_data_generator(self, batch_size=128):
        images_file_name = self._file_list[0]
        labels_file_name = self._file_list[1]
        gennerator = self._generate_a_batch(images_file_name, labels_file_name, batch_size)
        return gennerator

    def get_test_data_generator(self, batch_size=128):
        images_file_name = self._file_list[2]
        labels_file_name = self._file_list[3]
        gennerator = self._generate_a_batch(images_file_name, labels_file_name, batch_size)
        return gennerator

    def get_a_batch_data(self, data_generator):
        if sys.version > '3':
            batch_img, batch_labels, status = data_generator.__next__()
        else:
            batch_img, batch_labels, status = data_generator.next()
        return batch_img, batch_labels, status
