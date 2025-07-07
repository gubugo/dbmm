#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model, Sequential, load_model

# from ipycanvas import canvas

print("%.1f %.2f" % (12.4, 5.56))

empty_tensor_1 = tf.random.Generator.from_seed(360).normal(stddev=1, shape=(2,5))
empty_tensor_2 = tf.zeros((5,0))
print(f"Empty Tensor 2: {empty_tensor_2}")

empty_tensor_3 = Concatenate(axis=0)([empty_tensor_1, empty_tensor_2])
print(f"Empty Tensor 1: {empty_tensor_1}")
print(f"Empty Tensor 3: {empty_tensor_3}")
print(f"Shape of Empty Tensor 1: {tf.zeros((X.shape[0],2)).size()}")
print(f"Shape of Empty Tensor 2: {empty_tensor_2.shape}")
print(f"Shape of Empty Tensor 3: {empty_tensor_3.shape}")
