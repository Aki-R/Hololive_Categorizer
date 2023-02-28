import tensorflow as tf
import keras
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import RandomRotation, RandomZoom, RandomFlip, \
    RandomTranslation, Rescaling
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

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
ROBOCO = 9
MEL = 10
AKIROSE = 11
HACHAMA = 12
MATSURI = 13
AQUA = 14
SHION = 15
AYAME = 16
CHOCO = 17
SUBARU = 18
AZKI = 19
MIO = 20
OKAYU = 21
FLARE = 22
WATAME = 23
TOWA = 24
LUNA = 25
LAMY = 26
NENE = 27
BOTAN = 28
POLKA = 29
LAPLUS = 30
LUI = 31
KOYORI = 32
CHLOE = 33
IROHA = 34
RISU = 35
MOONA = 36
AIRANI = 37
OLLIE = 38
ANYA = 39
REINE = 40
ZETA = 41
KAERA = 42
KOBO = 43
CALLIOPE = 44
INANIS = 45
GURA = 46
AMELIA = 47
IRYS = 48
SANA = 49
FAUNA = 50
KRONII = 51
MUMEI = 52
BAELZ = 53
COCO = 54
FRIENDA = 55
NODOKA = 56
KIARA = 57


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
PATH_ROBOCO = './Train_Data/roboco/*'
PATH_MEL = './Train_Data/mel/*'
PATH_AKIROSE = './Train_Data/akirose/*'
PATH_HACHAMA = './Train_Data/hachama/*'
PATH_MATSURI = './Train_Data/matsuri/*'
PATH_AQUA = './Train_Data/aqua/*'
PATH_SHION = './Train_Data/shion/*'
PATH_AYAME = './Train_Data/ayame/*'
PATH_CHOCO = './Train_Data/choco/*'
PATH_SUBARU = './Train_Data/subaru/*'
PATH_AZUKI = './Train_Data/azuki/*'
PATH_MIO = './Train_Data/mio/*'
PATH_OKAYU = './Train_Data/okayu/*'
PATH_FALRE = './Train_Data/flare/*'
PATH_WATAME = './Train_Data/watame/*'
PATH_TOWA = './Train_Data/towa/*'
PATH_LUNA = './Train_Data/luna/*'
PATH_LAMY = './Train_Data/lamy/*'
PATH_NENE = './Train_Data/nene/*'
PATH_BOTAN = './Train_Data/botan/*'
PATH_POLKA = './Train_Data/polka/*'
PATH_LAPLUS = './Train_Data/laplus/*'
PATH_LUI = './Train_Data/lui/*'
PATH_KOYORI = './Train_Data/koyori/*'
PATH_CHLOE = './Train_Data/chloe/*'
PATH_IROHA = './Train_Data/iroha/*'
PATH_RISU = './Train_Data/risu/*'
PATH_MOONA = './Train_Data/moona/*'
PATH_AIRANI = './Train_Data/airani/*'
PATH_OLLIE = './Train_Data/ollie/*'
PATH_ANYA = './Train_Data/anya/*'
PATH_REINE = './Train_Data/reine/*'
PATH_ZETA = './Train_Data/zeta/*'
PATH_KAERA = './Train_Data/kaera/*'
PATH_KOBO = './Train_Data/kobo/*'
PATH_CALLIOPE = './Train_Data/calliope/*'
PATH_INANIS = './Train_Data/inanis/*'
PATH_GURA = './Train_Data/gura/*'
PATH_AMELIA = './Train_Data/amelia/*'
PATH_IRYS = './Train_Data/irys/*'
PATH_SANA = './Train_Data/sana/*'
PATH_FAUNA = './Train_Data/fauna/*'
PATH_KURONII = './Train_Data/kronii/*'
PATH_MUMEI = './Train_Data/mumei/*'
PATH_BAELZ = './Train_Data/baelz/*'
PATH_COCO = './Train_Data/coco/*'
PATH_FRIENDA = './Train_Data/frienda/*'
PATH_NODOKA = './Train_Data/nodoka/*'
PATH_KIARA = './Train_Data/kiara/*'

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


def ImageLoad(number, path):
    files = glob.glob(path)
    for file in files:
        image_pil = load_img(file, grayscale=False, color_mode='rgb',
                             target_size=(256, 256))
        image = np.array(image_pil, dtype=np.uint8)
        name = number
        images.append(image)
        names.append(name)


def LoadAllImage():
    ImageLoad(TOKINOSORA, PATH_TOKINOSORA)
    ImageLoad(PEKORA, PATH_PEKORA)
    ImageLoad(HUBUKI, PATH_HUBUKI)
    ImageLoad(MARIN, PATH_MARIN)
    ImageLoad(NOEL, PATH_NOEL)
    ImageLoad(SUISEI, PATH_SUISEI)
    ImageLoad(KORONE, PATH_KORONE)
    ImageLoad(MIKO, PATH_MIKO)
    ImageLoad(KANATA, PATH_KANATA)
    ImageLoad(ROBOCO, PATH_ROBOCO)
    ImageLoad(MEL, PATH_MEL)
    ImageLoad(AKIROSE, PATH_AKIROSE)
    ImageLoad(HACHAMA, PATH_HACHAMA)
    ImageLoad(MATSURI, PATH_MATSURI)
    ImageLoad(AQUA, PATH_AQUA)
    ImageLoad(SHION, PATH_SHION)
    ImageLoad(AYAME, PATH_AYAME)
    ImageLoad(CHOCO, PATH_CHOCO)
    ImageLoad(SUBARU, PATH_SUBARU)
    ImageLoad(AZKI, PATH_AZUKI)
    ImageLoad(MIO, PATH_MIO)
    ImageLoad(OKAYU, PATH_OKAYU)
    ImageLoad(FLARE, PATH_FALRE)
    ImageLoad(WATAME, PATH_WATAME)
    ImageLoad(TOWA, PATH_TOWA)
    ImageLoad(LUNA, PATH_LUNA)
    ImageLoad(LAMY, PATH_LAMY)
    ImageLoad(NENE, PATH_NENE)
    ImageLoad(BOTAN, PATH_BOTAN)
    ImageLoad(POLKA, PATH_POLKA)
    ImageLoad(LAPLUS, PATH_LAPLUS)
    ImageLoad(LUI, PATH_LUI)
    ImageLoad(KOYORI, PATH_KOYORI)
    ImageLoad(CHLOE, PATH_CHLOE)
    ImageLoad(IROHA, PATH_IROHA)
    ImageLoad(RISU, PATH_RISU)
    ImageLoad(MOONA, PATH_MOONA)
    ImageLoad(AIRANI, PATH_AIRANI)
    ImageLoad(OLLIE, PATH_OLLIE)
    ImageLoad(ANYA, PATH_ANYA)
    ImageLoad(REINE, PATH_REINE)
    ImageLoad(ZETA, PATH_ZETA)
    ImageLoad(KAERA, PATH_KAERA)
    ImageLoad(KOBO, PATH_KOBO)
    ImageLoad(CALLIOPE, PATH_CALLIOPE)
    ImageLoad(INANIS, PATH_INANIS)
    ImageLoad(GURA, PATH_GURA)
    ImageLoad(AMELIA, PATH_AMELIA)
    ImageLoad(IRYS, PATH_IRYS)
    ImageLoad(SANA, PATH_SANA)
    ImageLoad(FAUNA, PATH_FAUNA)
    ImageLoad(KRONII, PATH_KURONII)
    ImageLoad(MUMEI, PATH_MUMEI)
    ImageLoad(BAELZ, PATH_BAELZ)
    ImageLoad(COCO, PATH_COCO)
    ImageLoad(FRIENDA, PATH_FRIENDA)
    ImageLoad(NODOKA, PATH_NODOKA)
    ImageLoad(KIARA, PATH_KIARA)


if __name__ == "__main__":
    # Disable GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Show Tensorflow and Keras version
    print(tf.__version__)
    print(keras.__version__)

    LoadAllImage()

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
