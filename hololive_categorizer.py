import tensorflow as tf
import keras
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import RandomRotation, RandomZoom, RandomFlip, \
    RandomTranslation, Rescaling, Input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd

images = []
names = []


def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    X_train, X_valid, y_train, y_valid = \
        train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
    # Setting model
    opt = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint(filepath='model_best.h5', monitor='val_loss',
                         verbose=1, save_best_only=True,
                         save_weights_only=False, mode='min', period=1)
    # Learning model
    fit = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    validation_data=(X_valid, y_valid), callbacks=[cp])
    # Saving model
    model.save('model.h5')
    return fit


def vgg16_model():
    # base vgg16 model
    conv_base = VGG16(weights='imagenet', include_top=False)
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(Rescaling(scale=1./255, input_shape=(256, 256, 3)))
    model.add(RandomRotation(factor=0.15))
    model.add(RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)))
    model.add(RandomFlip('horizontal'))
    model.add(RandomTranslation(height_factor=0.2, width_factor=0.2))
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(58, activation='softmax'))
    return model


def vgg16_functinal_model():
    input = Input(shape=(256, 256, 3))
    x = Rescaling(scale=1./255, input_shape=(256, 256, 3))(input)
    x = RandomRotation(factor=0.15)(x)
    x = RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))(x)
    x = RandomFlip("horizontal")(x)
    x = RandomTranslation(height_factor=0.2, width_factor=0.2)(x)

    conv_base = VGG16(weights='imagenet', include_top=False, input_tensor=x)
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    x = Flatten()(conv_base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(58, activation='softmax')(x)
    model = keras.Model(inputs=input, outputs=x)
    return model


def ImageLoad(number, path):
    files = glob.glob(path)
    for file in files:
        image_pil = load_img(file, grayscale=False, color_mode='rgb',
                             target_size=(256, 256))
        image = np.array(image_pil, dtype=np.uint8)
        name = number
        images.append(image)
        names.append(name)


if __name__ == "__main__":
    # Disable GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Show Tensorflow and Keras version
    print(tf.__version__)
    print(keras.__version__)

    df = pd.read_csv("vtuber_list.csv")

    for index, row in df.iterrows():
        ImageLoad(row["id"], row["path"])

    # Number to category array
    images_np = np.array(images)
    y_names = to_categorical(names)

    print('size is ........................')
    print(images_np.shape)
    print(y_names)

    epochs = 20
    batch_size = 128

    model = vgg16_model()
    fit = trainModelAndSave(model, images_np, y_names, epochs, batch_size)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(fit.history['loss'])
    axs[0].plot(fit.history['val_loss'])
    axs[0].set_ylabel('crossentropy loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['training data', 'validation data'], loc='upper right')

    axs[1].plot(fit.history['accuracy'])
    axs[1].plot(fit.history['val_accuracy'])
    axs[1].set_ylabel('crossentropy accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['training data', 'validation data'], loc='lower right')

    plt.show()
    plt.close(fig)
