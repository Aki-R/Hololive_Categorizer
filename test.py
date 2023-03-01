import random
import cv2
import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img

MODEL_PATH = './model.h5'
PATH_TOKINOSORA = './Train_Data/tokinosora/*'
PATH_PEKORA = './Train_Data/pekora/*'
PATH_HUBUKI = './Train_Data/hubuki/*'

files = glob.glob(PATH_PEKORA)
file = random.choice(files)
print(file)
img = cv2.imread(file, 1)
cv2.imshow('original image', img)

model = load_model(MODEL_PATH)

image_pil = load_img(file, grayscale=False, color_mode='rgb', target_size=(256, 256))
image = np.array(image_pil, dtype=np.uint8)
image_flip = cv2.flip(image, 1)
new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow('reshaped image', new_image)

new_image_flip = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
cv2.imshow('flip image', new_image_flip)
image_expand = image[np.newaxis, :, :, :]
print(image.shape)
print(image_expand.shape)

result = model.predict(image_expand, batch_size=1)
print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()
