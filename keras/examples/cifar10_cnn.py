# coding=utf-8
from keras.datasets import  cifar10
from keras.layers import Dropout,Dense,Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.image import  ImageDataGenerator
import os

batch_size = 128
epoches = 10
num_classes = 10
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'save_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print('# of train samples:', x_train.shape[0])
print('# of test samples:', x_test.shape[0])
# print('x_train data:', x_train)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=x_train[0].shape))
model.add(Conv2D(32,(3,3), padding='SAME', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
# model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
# model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

opt = RMSprop(0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

if not data_augmentation:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoches, shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip imagesdatagen.fit(x_train)
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), shuffle=False, workers=4, epochs=epoches)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])





