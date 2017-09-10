# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:42:34 2017

@author: kopyt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model import LinearModel

def true_function(x, noise = True):
    y = -5*x+5
    if noise:
        y += np.random.normal(scale=0.1, size = x.shape)
    return y

EPOCHS = 100
R = 200

X_data = np.arange(-5,5,0.5).reshape((-1,1))
Y_data = true_function(X_data) # linear function
#perturbation
Y_data[1] = 100.0
Y_data[-5] = 100.0

Y_data = Y_data.reshape((-1,1))

X_test = 3.2*np.ones((1,1))
Y_test = true_function(X_test, noise=False)

num_train_points = X_data.shape[0]

tf.reset_default_graph()

x = tf.placeholder(dtype=tf.float32, shape=(None, 1))
y_true = tf.placeholder(dtype=tf.float32, shape=(None, 1))

model = LinearModel(x, y_true)

train_op = model.optimize
loss_op = model.error
param_op = model.params
gradient_op = model.gradients
hessian_op = model.hessians

init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    
    sess.run(init_op)
    
    for e in range(EPOCHS):
        
        fd = {x: X_data, y_true: Y_data}
        _, loss_epoch = sess.run([train_op, loss_op], feed_dict = fd)
        
    p = sess.run(param_op)
    
    
    
    s_test = 0
    for r in range(R):
        
        v = sess.run(gradient_op, feed_dict = {x:X_test, y_true:Y_test})[0]
        s_test_j = v
        for j in range(num_train_points):
            fd = {x:X_data[j].reshape((-1,1)), y_true:Y_data[j].reshape((-1,1))}
            hess_param = sess.run(hessian_op, feed_dict = fd)[0]
            hess_param = np.diag(hess_param)
            s_test_j = v + np.matmul((np.identity(2)-hess_param),s_test_j)
            
        s_test += s_test_j
        
    s_test = s_test/R
    
    importance = []
    for j in range(num_train_points):
        fd = {x:X_data[j].reshape((-1,1)), y_true:Y_data[j].reshape((-1,1))}
        grad_param = sess.run(gradient_op, feed_dict = fd)[0]
        importance.append(-np.matmul(s_test,grad_param))
        
print('Loss',(p[0]*X_test+p[1]-Y_test)**2)

importance = np.asarray(importance)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(X_data, Y_data,label='train',c=-importance,cmap=cm)
plt.scatter(X_test, Y_test,marker='+',label='test',c='r')
plt.plot(X_data, p[0]*X_data+p[1])
plt.colorbar(sc)
plt.show()
    