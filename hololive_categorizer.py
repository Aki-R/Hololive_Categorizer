import tensorflow as tf
import keras
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import RandomRotation, RandomZoom, RandomFlip, RandomTranslation, Rescaling
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Show Tensorflow and Keras version
print(tf.__version__)
print(keras.__version__)

# Categorize No.
TOKINOSORA = 0
PEKORA = 1
HUBUKI = 2
MARIN = 3
NOEL = 4
SUISEI = 5
KORONE = 6
MIKO = 7
KANATA = 8

HOLOLIST = ['ときのそら','うさだぺこら','白上ふぶき','宝鐘マリン','白銀ノエル','星街すいせい','戌神ころね']

# Image path
PATH_TOKINOSORA = './Train_Data/tokinosora/*'
PATH_PEKORA = './Train_Data/pekora/*'
PATH_HUBUKI = './Train_Data/hubuki/*'
PATH_MARIN = './Train_Data/marin/*'
PATH_NOEL = './Train_Data/noel/*'
PATH_SUISEI = './Train_Data/suisei/*'
PATH_KORONE = './Train_Data/korone/*'
PATH_MIKO = './Train_Data/miko/*'
PATH_KANATA = './Train_Data/kanata/*'

def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
    # Setting model
    #opt = tf.keras.optimizers.RMSprop(learning_rate=1.0e-4)
    opt = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # Learning model
    fit = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))
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
    model.add(Dense(9, activation='softmax'))
    return model

def ImageLoad(number, path):
    files = glob.glob(path)
    for file in files:
        image_pil = load_img(file, grayscale=False, color_mode='rgb', target_size=(256, 256))
        image = np.array(image_pil, dtype=np.uint8)
        name = number
        images.append(image)
        names.append(name)

images = []
names = []

ImageLoad(TOKINOSORA, PATH_TOKINOSORA)
ImageLoad(PEKORA, PATH_PEKORA)
ImageLoad(HUBUKI, PATH_HUBUKI)
ImageLoad(MARIN, PATH_MARIN)
ImageLoad(NOEL, PATH_NOEL)
ImageLoad(SUISEI, PATH_SUISEI)
ImageLoad(KORONE, PATH_KORONE)
ImageLoad(MIKO, PATH_MIKO)
ImageLoad(KANATA, PATH_KANATA)


# Number to category array
images_np = np.array(images)
y_names = to_categorical(names)

print('size is ........................')
print(images_np.shape)
print(y_names)

epochs = 30
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
