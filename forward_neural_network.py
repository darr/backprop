#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : forward_neural_network.py
# Create date : 2018-12-25 20:04
# Modified date : 2018-12-28 14:05
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import numpy as np
import mnist_dataset
import matplotlib.pyplot as plt

class DeepModel(object):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.hyper = self._get_hyperparameters()

    def _get_hyperparameters(self):
        dic = {}
        dic["batch_size"] = 128
        dic["epsilon"] = 0.0000001
        dic["reg_lambda"] = 0.05
        dic["learn_rate"] = 0.001
        dic["max_steps"] = 500
        dic["train_steps"] = 1
        dic["max_epochs"] = 20
        dic["input_dim"] = 28*28 #img size
        dic["hidden_dim"] = 2048
        dic["output_dim"] = 10
        dic["file_path"] = "./data/"
        return dic

    def show_hyperparameters(self):
        print("pyperparameters:")
        for key in self.hyper:
            print("%s:%s" % (key, self.hyper[key]))

    def _sig(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def _relu(self, x):
        return (np.abs(x) + x) / 2.0

    def _relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def _sig_deirvative(self, x):
        return self._sig(x) * (1 - self._sig(x))

    def _deal_softmax_numerical(self, logits):
        logits_max = np.max(logits, axis=1)
        for i in range(len(logits)):
            logits[i] = logits[i] - logits_max[i]
        logits = logits + self.hyper["epsilon"]
        return logits

    def _deal_log_numerical(self, probs):
        probs = probs + self.hyper["epsilon"]
        return probs

    def _weight_decay(self, dW2, dW1, W2, W1):
        dW2 += self.hyper["reg_lambda"] * W2
        dW1 += self.hyper["reg_lambda"] * W1
        return dW1, dW2

    def _init_model(self, model=None):
        if model:
            W1 = model["W1"]
            b1 = model["b1"]
            W2 = model["W2"]
            b2 = model["b2"]
        else:
            np.random.seed(0)
            W1 = np.random.randn(self.hyper["input_dim"], self.hyper["hidden_dim"])
            b1 = np.ones((1, self.hyper["hidden_dim"]))
            W2 = np.random.randn(self.hyper["hidden_dim"], self.hyper["output_dim"])
            b2 = np.ones((1, self.hyper["output_dim"]))

            model = {}
            model["W1"] = W1
            model["b1"] = b1
            model["W2"] = W2
            model["b2"] = b2
            model["steps"] = 1
            model["epochs"] = 0
            model["hyperparameters"] = self.hyper
            model["record"] = {}
        return model, W1, b1, W2, b2

    def _normalization(self, batch_img):
        return batch_img / 255.

    def _get_data(self, dataset, data_generator):
        batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
        X = batch_img
        X = self._normalization(X)
        Y = batch_labels
        return X, Y, status

    def _print_train_status(self, model):
        print("epoch:%s steps:%s Train_Loss:%2.5f Train_Acc:%2.5f" % (model["epochs"], model["steps"], model["train_loss"], model["train_accuracy"]))

    def _print_test_status(self, model):
        print("epoch:%s steps:%s Train_Loss:%2.5f Test_Loss:%2.5f Train_Acc:%2.5f Test_Acc:%2.5f train_test_gap:%2.5f" % (model["epochs"], model["steps"], model["train_loss"], model["test_loss"], model["train_accuracy"], model["test_accuracy"], model["train_test_gap"]))

    def _forward_propagation(self, model, X, Y):
        model, W1, b1, W2, b2 = self._init_model(model)

        Z1 = np.dot(X, W1)+b1
        a1 = self._relu(Z1)
        #a1 = self._sig(Z1)
        logits = np.dot(a1, W2)+b2
        logits = self._deal_softmax_numerical(logits)
        exp_score = np.exp(logits)
        prob = exp_score/np.sum(exp_score, axis=1, keepdims=1)

        correct_probs = prob[range(X.shape[0]), np.argmax(Y, axis=1)]
        correct_probs = self._deal_log_numerical(correct_probs)
        correct_logprobs = -np.log(correct_probs)

        data_loss = np.sum(correct_logprobs)
        loss = 1./X.shape[0] * data_loss

        pre_Y = np.argmax(prob, axis=1)
        comp = pre_Y == np.argmax(Y, axis=1)
        accuracy = len(np.flatnonzero(comp))/Y.shape[0]

        return model, prob, a1, Z1, loss, accuracy, comp

    def _backward_propagation(self, model, prob, X, Y, a1, Z1):
        W1 = model["W1"]
        W2 = model["W2"]
        dY_pred = prob - Y
        dW2 = np.dot(a1.T, dY_pred)
        da1 = np.dot(dY_pred, W2.T)
        dadZ = self._relu_derivative(Z1)
        #dadZ = self._sig_deirvative(Z1)
        dZ1 = da1 * dadZ
        dW1 = np.dot(X.T, dZ1)
        #dW1,dW2 = self.weight_decay(dW2,dW1,W2,W1)
        model["W2"] += -self.hyper["learn_rate"]*dW2
        model["W1"] += -self.hyper["learn_rate"]*dW1

        return model

    def _core_graph(self, model, X, Y):
        model, prob, a1, Z1, loss, accuracy, comp = self._forward_propagation(model, X, Y)
        model["train_loss"] = loss
        model["train_accuracy"] = accuracy
        model = self._backward_propagation(model, prob, X, Y, a1, Z1)
        return model

    def _train_model_with_epochs(self, model=None):
        dataset = mnist_dataset.MnistSet(self.hyper["file_path"])
        data_generator = dataset.get_train_data_generator(self.hyper["batch_size"])
        while 1:
            X, Y, status = self._get_data(dataset, data_generator)
            if status == False:
                model['epochs'] += 1
                break
            model = self._core_graph(model, X, Y)
            model["steps"] += 1
            if model["steps"] % self.hyper["train_steps"] == 0:
                self._print_train_status(model)

        return model

    def _train_model_with_steps(self, model=None, data_generator=None):
        dataset = mnist_dataset.MnistSet(self.hyper["file_path"])
        if data_generator == None:
            data_generator = dataset.get_train_data_generator(self.hyper["batch_size"])
        while 1:
            X, Y, status = self._get_data(dataset, data_generator)
            if status == False:
                data_generator = dataset.get_train_data_generator(self.hyper["batch_size"])
                model['epochs'] += 1
            model = self._core_graph(model, X, Y)

            model["steps"] += 1
            if model["steps"] % self.hyper["train_steps"] == 0:
                break

        return model, data_generator

    def _test_update_model(self, model, avg_loss, accuracy):
        model["test_loss"] = avg_loss
        model["test_accuracy"] = accuracy
        model["train_test_gap"] = model["train_accuracy"] - model["test_accuracy"]
        return model

    def _test_model(self, model):
        dataset = mnist_dataset.MnistSet(self.hyper["file_path"])
        data_generator = dataset.get_test_data_generator(self.hyper["batch_size"])
        count = 1
        all_correct_numbers = 0
        all_loss = 0.0

        while count:
            X, Y, status = self._get_data(dataset, data_generator)
            if status == False:
                break
            model, prob, a1, Z1, loss, accuracy, comp = self._forward_propagation(model, X, Y)
            all_loss += loss
            all_correct_numbers += len(np.flatnonzero(comp))
            count += 1

        avg_loss = all_loss / count
        accuracy = all_correct_numbers / 10000.0
        self._test_update_model(model, avg_loss, accuracy)
        self._print_test_status(model)
        self._record_model_status(model)
        return model

    def _record_model_status(self, model):
        steps_dic = {}
        steps_dic["epochs"] = model["epochs"]
        steps_dic["steps"] = model["steps"]
        steps_dic["train_loss"] = model["train_loss"]
        steps_dic["train_accuracy"] = model["train_accuracy"]
        steps_dic["test_loss"] = model["test_loss"]
        steps_dic["test_accuracy"] = model["test_accuracy"]
        steps_dic["train_test_gap"] = model["train_test_gap"]
        record = model["record"]
        record[model["steps"]] = steps_dic

    def _plot_record(self, model):
        self._plot_a_key(model, "train_loss", "test_loss")
        self._plot_a_key(model, "train_accuracy", "test_accuracy")

    def _plot_a_key(self, model, train_key, test_key):
        record = model["record"]
        train = []
        test = []
        steps = []
        for key in record:
            steps.append([key])
        steps.sort()
        for i in range(len(steps)):
            step_dic = record[steps[i][0]]
            train_value = step_dic[train_key]
            train.append(train_value)
            test_value = step_dic[test_key]
            test.append(test_value)
        train = np.array(train)
        steps = np.array(steps)
        plt.plot(steps, train)
        plt.plot(steps, test)
        plt.show()

    def run_with_epoch(self):
        model = None
        while 1:
            model = self._train_model_with_epochs(model)
            self._test_model(model)
            if model["epochs"] > self.hyper["max_epochs"]:
                break
        self._plot_record(model)

    def run_with_steps(self):
        model = None
        data_generator = None
        while 1:
            model, data_generator = self._train_model_with_steps(model, data_generator)
            model = self._test_model(model)
            if model["steps"] > self.hyper["max_steps"]:
                break
        self._plot_record(model)
