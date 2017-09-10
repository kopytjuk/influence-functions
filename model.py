# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 08:53:15 2017

@author: kopyt
"""
import tensorflow as tf

class LinearModel:
    
    def __init__(self, data, target):
        
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None
        self._gradients = None
        self._hessians = None
        self._params = None
        
    @property
    def params(self):
        if self._params is None:
            data_dim = int(self.data.get_shape()[1])
            target_dim = int(self.target.get_shape()[1])
            # we construct one variable for both weight and bias
            self._params = tf.get_variable(name='params', shape=[data_dim+target_dim])
        return self._params
        
    @property
    def prediction(self):
        if self._prediction is None:
            data_dim = int(self.data.get_shape()[1])
            W = tf.reshape(self.params[:-1], [data_dim,1])
            b = self.params[-1]
            self._prediction = tf.matmul(self.data, W) + b
        return self._prediction
    
    @property
    def error(self):
        if self._error is None:
            self._error = tf.losses.mean_squared_error(labels = self.target, 
                                                       predictions = self.prediction)
        return self._error
    
    @property
    def optimize(self, lr = 0.1):
        if self._optimize is None:
            train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.error)
            self._optimize = train_op
        return self._optimize
    
    @property
    def gradients(self):
        if self._gradients is None:
            self._gradients = tf.gradients(self.error, self.params)
        return self._gradients
    
    @property
    def hessians(self):
        if self._hessians is None:
            self._hessians = tf.hessians(self.error, self.params)
        return self._hessians