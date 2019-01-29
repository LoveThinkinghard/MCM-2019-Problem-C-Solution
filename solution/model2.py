# -*- coding: utf-8 -*-
# if you run 'model1.py' just now, you need to restart your kernel. 
# or you might have some trouble when you fit the model

import keras.layers as kl
from keras import Model
import numpy as np
import matplotlib.pyplot as plt
#%%
dis_mat = np.load('./data/distance_matrix.npy')
drug_use = np.load('./data/drug_use.npy')
all_s_use = np.load('./data/all_s_use.npy')

# this eco_data comes from socio-economic data, which is difficult to pre-process
# and actually, we made some mistakes on this thing 
# which means that the data below is wrong, but can be used
# if you want to, you can try it yourself
x_eco_data = np.load('./data/eco_data.npy')
#%%
X = []
Y = []
for i in range(7):
    for n in range(28):
        if all_s_use[n, i].sum()>0:
            X.append(list([np.matmul(all_s_use[n, i], dis_mat)])+list(x_eco_data[i]))
            Y.append(all_s_use[n, i+1])
X3 = []
for i in range(462):
    X3.append([X[n][i] for n in range(123)])
#%%
counties_input = kl.Input(shape=(461,))
eco_inputs = []
eco_mat = []
shared_dense = kl.Dense(1)

for i in range(461):
    eco_inputs.append(kl.Input(shape=(197,)))
    eco_mat.append(shared_dense(eco_inputs[-1]))

eco_mat = kl.concatenate(eco_mat)
hide_input = kl.multiply([counties_input, eco_mat])
output = kl.Dense(461)(hide_input)
 
model = Model([counties_input]+eco_inputs, output)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
#%%
# it takes several minutes, and you will not go faster even you use 1080ti, I tried
# it might because we use a shared Dense layer for 461 inputs
history = model.fit(x=X3, y=[Y], batch_size=4, epochs=200)
#%%
plt.plot(history.epoch, history.history['loss'])
#%%
eco_weight = model.get_weights()[0]
plt.plot(range(eco_weight.size), eco_weight)
#%%
# head are tags of socio-economic data in a certain order
# this just where we do wrong, because different tags are used in each file.
head = np.load('./data/head.npy')
order = eco_weight.argsort(axis=0)
one_node = model.get_weights()[1]
print(head[order])
print('this list is in a increase order')
if one_node>0:
    print('larger the parameter, more the drug use')
else:
    print('larger the parameter, less the drug use')
