# -*- coding: utf-8 -*-

import keras.layers as kl
from keras import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%%
# load and display distance_matrix, which is designed by another teammate.
# you can download it directly, or make you own design
dis_mat = np.load('./data/distance_matrix.npy')

plt.imshow(dis_mat)
plt.colorbar()
#%%
# files below can be generated in the last part of 'plot_gif.py'
t = np.load('./data/county.npy')
drug_use = np.load('./data/drug_use.npy')
all_s_use = np.load('./data/all_s_use.npy').astype('float32')
#%%
x_data = all_s_use[0][0:-1]
y_data = all_s_use[0][1:]
for i in range(27):
    x_data = np.concatenate((x_data, all_s_use[i+1][0:-1]), axis=0)
    y_data = np.concatenate((y_data, all_s_use[i+1][1:]), axis=0)
print(x_data.shape)

# clear some zero data out
cut = np.array([x_data[i].sum()>0 for i in range(196)])
x_data = x_data[cut]
y_data = y_data[cut]
print(x_data.shape)
#%%
# this is where model begin
model = Sequential()
model.add(kl.Dense(461, input_shape=(461,)))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
#%%
history = model.fit(x=np.matmul(x_data, dis_mat), y=y_data, batch_size=8, epochs=200)
#%%
plt.plot(history.epoch, history.history['loss'])
#%%
# make prediction
x = [21, 39, 42, 51, 54]
m = 11 # Fentanyl
fig, ax = plt.subplots(5, 1, sharex=True)
y_predict = model.predict(np.matmul(all_s_use[m], dis_mat))
y_2 = [model.predict(np.matmul([y_predict[-1]], dis_mat))]
for _ in range(3):
    y_2.append(model.predict(np.matmul(y_2[-1], dis_mat)))
y_predict = np.concatenate((y_predict, np.array(y_2).reshape(4, 461)), axis=0)
ax[0].set_title('{} of 5 states in five years'.format(drug_use[m]))
for n in range(5):
    cut = np.where(((t/1000).astype(int) == x[n]))
    ax[n].plot([2010+i for i in range(8)], [all_s_use[m][i][cut].sum() for i in range(8)])
    ax[n].plot([2011+i for i in range(12)], [y_predict[i][cut].sum() for i in range(12)])
ax[0].legend(['original', 'prediction'])
ax[4].set_xlabel('year')
ax[0].set_ylabel('drug report number')
fig.savefig('./prediction/{} of 5 states in five years'.format(drug_use[m]))

#%%
# below is a inverse model to identify where specific opioid use started
# however maybe we do something wrong, it doesn't work actually
# hope you can figure out why. And if you do, please tell us. Thanks
#%%
weights = model.get_weights()
cor_mat = weights[0]
cor_bias = weights[1]
def inverse(x):
    y = np.matmul(x-cor_bias, np.linalg.inv(cor_mat))
    return np.matmul(y, np.linalg.inv(dis_mat))
#%%
x = [21, 39, 42, 51, 54]
m= 11
fig, ax = plt.subplots(5, 1, sharex=True)
y_1 = []
for i in range(8):
    y_1.append(inverse(all_s_use[m][7-i]))
for i in range(4):
    y_1.append(inverse(y_1[-1]))
ax[0].set_title('{} of 5 states inverse 5 years'.format(drug_use[m]))
for n in range(5):
    cut = np.where(((t/1000).astype(int) == x[n]))
    ax[n].plot([2010+i for i in range(8)], [all_s_use[m][i][cut].sum() for i in range(8)])
    ax[n].plot([2017-i for i in range(12)], [y_1[i][cut].sum() for i in range(12)])
ax[0].legend(['original', 'prediction'])
ax[4].set_xlabel('year')
ax[0].set_ylabel('drug report number')

#%%
# tip
#%%
# if you need to remake a model run this
tf.reset_default_graph()