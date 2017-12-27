# coding=utf-8
from PIL import Image
import os
import numpy as np
import pandas as pd
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        print("1")
        #images = images.astype(np.float32)
        print("2")
        #images = np.multiply(images, 1.0 / 255.0)
        print("3")
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 1024*1024
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, label):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = len(labels_dense)
  index_offset = list(labels_dense).index(label)
  labels_one_hot = np.zeros(num_labels)
  labels_one_hot[index_offset] = 1
  return labels_one_hot


print(dense_to_one_hot(['a','b','ccc','dddd'],'ccc'))
def read_data_sets():
    image_path="/Users/hanst/Documents/文档/医疗AI/NIH dataset/images/image001"
    label_path="/Users/hanst/Documents/文档/医疗AI/NIH dataset/Data_Entry_2017.csv"

    data_label = pd.read_csv(label_path)
    data_arr = data_label.ix[:,['Image Index', 'Finding Labels']]
    image_labels = {}
    num_data=len(data_arr)
    for i in range(num_data):
        image_name = data_arr.loc[i]['Image Index']
        image_label = data_arr.loc[i]['Finding Labels'].split('|')[0]
        image_labels[image_name]=image_label
        if i%10000==0:
            print("image_name, image_label:", image_name,image_label)

    uniq_labels = set(image_labels.values())
    print("# of Unique labels:",len(uniq_labels))


    images = []
    file_list = []
    labels =[]
    i = 0
    for file in os.listdir(image_path):
        if file.endswith("png"):
            im = Image.open(image_path+"/"+file)
            image_array = np.array(im)
            image_array.astype(np.float32)
            image_array = np.multiply(image_array, 1.0 / 255.0)
            label = image_labels.get(file)

            if i == 0:
                #im.show()
                print("image shape:", image_array.shape, image_array)
            if i%1000==0:
                print("file path, label:", str(file), label)

            if image_array.shape == (1024,1024):
                images.append(image_array)
                i += 1
                file_list.append(file)
                labels.append(dense_to_one_hot(uniq_labels, label))
                if(i>2000): break
            else:
                print("file path,image shape:", str(file), image_array.shape)
                #images.append(image_array[:,:,1])

    length = len(file_list)
    np_images = np.reshape(np.array(images), (length,1024*1024))
    np_labels = np.reshape(np.array(labels),(length, 15))
    print("image #:", length)
    print("image shape:", np_images.shape)
    print("labels shape:", np_labels.shape)
    images = []
    labels = []
    train_images = np_images[0:1500]
    train_labels = np_labels[0:1500]

    test_images = np_images[1500:]
    test_labels = np_labels[1500:]
    np_images=[]
    np_labels=[]
    print("build dataset")
    train = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)
    return (train, test)





