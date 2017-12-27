# coding=utf-8
# coding=utf-8
from PIL import Image
import os
import numpy as np
import pandas as pd
from keras.preprocessing import  image

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
            im = image.load_img(image_path+"/"+file, grayscale=False, target_size=(512,512))
            image_array = image.img_to_array(im)
            image_array = np.multiply(image_array, 1.0 / 255.0)
            label = image_labels.get(file)

            if i == 0:
                #im.show()
                print("image shape:", image_array.shape, image_array)
            if i%1000==0:
                print("file path, label:", str(file), label)

             #print("file path,image shape:", str(file), image_array.shape)
            images.append(image_array)
            i += 1
            file_list.append(file)
            labels.append(dense_to_one_hot(uniq_labels, label))
            # if i>10: break
    length = len(file_list)
    np_images = np.reshape(np.array(images), (length,512,512,3))
    np_labels = np.reshape(np.array(labels),(length, 15))
    print("image #:", length)
    print("image shape:", np_images.shape)
    print("labels shape:", np_labels.shape)
    perm0 = np.arange(length)
    np.random.shuffle(perm0)
    np_images= np_images[perm0]
    np_labels = np_labels[perm0]
    train_images = np_images[0:4000]
    train_labels = np_labels[0:4000]

    test_images = np_images[4000:]
    test_labels = np_labels[4000:]
    print("build dataset")
    return (train_images,train_labels, test_images,test_labels)

def read_and_save(clazz):
    print('read_and_save, class label:',clazz)
    image_path = "/Users/hanst/Documents/文档/医疗AI/NIH dataset/images/image001"
    label_path = "/Users/hanst/Documents/文档/医疗AI/NIH dataset/Data_Entry_2017.csv"
    base_path = '/Users/hanst/Documents/文档/医疗AI/NIH dataset/new_images'
    data_label = pd.read_csv(label_path)
    data_arr = data_label.ix[:, ['Image Index', 'Finding Labels']]
    image_labels = {}
    num_data = len(data_arr)
    for i in range(num_data):
        image_name = data_arr.loc[i]['Image Index']
        image_label = data_arr.loc[i]['Finding Labels']
        image_labels[image_name] = image_label
        if i % 10000 == 0:
            print("image_name, image_label:", image_name, image_label)

    uniq_labels = set(image_labels.values())
    print("# of Unique labels:", len(uniq_labels))

    for file in os.listdir(image_path):
        if file.endswith("png"):
            im = image.load_img(image_path+"/"+file, grayscale=False, target_size=(512,512))

            labels = image_labels.get(file)

            if clazz in labels:
                print('file, labels:', file, labels)
                new_path = os.path.join(base_path,clazz)
                if not os.path.isdir(new_path):
                    os.makedirs(new_path)
                im.save(os.path.join(new_path,file))
                print('new file saved:', new_path, file)


# read_and_save('Cardiomegaly')
# read_and_save('No Finding')





