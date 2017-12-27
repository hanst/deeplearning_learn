# coding=utf-8
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

from keras.datasets import reuters
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#
# Fashion-MNIST数据集
# 本数据集包含60,000个28x28灰度图像，共10个时尚分类作为训练集。测试集包含10,000张图片。该数据集可作为MNIST数据集的进化版本，10个类别标签分别是：
#
# 类别	描述
# 0	T恤/上衣
# 1	裤子
# 2	套头衫
# 3	连衣裙
# 4	大衣
# 5	凉鞋
# 6	衬衫
# 7	帆布鞋
# 8	包
# 9	短靴
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Boston房屋价格回归数据集
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
