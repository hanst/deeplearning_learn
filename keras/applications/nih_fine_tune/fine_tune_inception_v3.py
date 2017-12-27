# coding=utf-8
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import datetime

import nih_image_input


def build_model(base_model):
    # add a global spatial average pooling layer
    """
    build refine model
    :param base_model:
    :return:
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 15 classes
    predictions = Dense(15, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions, name='nih_disease_model')
    return model


def first_compile_train(base_model, new_model, datagen):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # train the model on the new data for a few epochs
    new_model.fit_generator(datagen.flow(train_images, train_labels), epochs=3)


def fine_tune_model(base_model:Model, model: Model, num_epochs, datagen):
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    filepath = 'weights/' +model.name +'.'+date+ '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
                                 mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit_generator(datagen.flow(train_images, train_labels), epochs=num_epochs,callbacks=[checkpoint, tensor_board])
    return history

first_train = False
weights_name = 'weights/nih_nih_disease_model.10-1.60.hdf5'
train_images, train_labels,test_images, test_labels = nih_image_input.read_data_sets()
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

# create the base pre-trained model


if first_train:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = build_model(base_model)
    first_compile_train(base_model,model, datagen)
    fine_tune_model(base_model, model, 10, datagen)
else:
    base_model = InceptionV3(weights=None, include_top=False)
    model = build_model(base_model)
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.load_weights(weights_name)
    model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    score = model.evaluate(test_images, test_labels, batch_size=10)
    print('Test loss for last stored weights:', score)
    fine_tune_model(base_model, model, 10, datagen)

score = model.evaluate(test_images, test_labels, batch_size=10)
print('Test loss:', score)

model.load_weights('weights/nih_disease_model.01-11.69.hdf5')
score = model.evaluate(test_images, test_labels, batch_size=32)
print('Test loss:', score)

model.load_weights('weights/nih_disease_model.03-11.69.hdf5')
score = model.evaluate(test_images, test_labels, batch_size=32)
print('Test loss:', score)

model.load_weights('weights/nih_disease_model.04-11.69.hdf5')
score = model.evaluate(test_images, test_labels, batch_size=32)
print('Test loss:', score)
