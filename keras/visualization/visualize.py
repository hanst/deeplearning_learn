# coding=utf-8

from functional_model import  share_vision_module
from keras.utils import  plot_model
import numpy as np
from keras.optimizers import SGD

model = share_vision_module.build_model()


input_a = np.random.random((10, 256,256, 3))
input_b = np.random.random((10, 256,256, 3))
y_ = np.random.random((10,1))
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=[input_a, input_b], y = y_, batch_size=10, epochs=1)
plot_model(model, to_file='share_vison_modele.png', show_layer_names=True, show_shapes=True)
