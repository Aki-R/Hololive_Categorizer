import tkinter as tk
from tkinter import ttk
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import ImageGrab
import cv2
import numpy as np

MODEL_PATH = './model.h5'

HOLOLIST = ['ときのそら','うさだぺこら','白上ふぶき','宝鐘マリン','白銀ノエル','星街すいせい','戌神ころね','さくらみこ','天音かなた']

# Global Variable
image = {}

class Frame_Control(tk.Frame):
    def __init__(self, master=None):
        # Windowの初期設定を行う。
        super().__init__(master)
        self.master.wm_attributes("-transparentcolor", "snow")
        self.master.geometry("256x300")
        self.master.resizable(width=False, height=False)
        self.master.title("Hololive Categorizer")
        self.configure(width=256, height=44)
        self.pack()
        self.create_wigdgets()
        # initialize Keras model
        self.model = load_model(MODEL_PATH)

    def create_wigdgets(self):
        #Start Button
        self.button_start = ttk.Button(self)
        self.button_start.configure(text='Start')
        self.button_start.configure(command = self.start_categorize)
        self.button_start.pack()

        #Text Display for Result
        self.label_result = tk.Label(self)
        self.label_result.configure(text='Who')
        self.label_result.pack()

    def start_categorize(self):
        #result = model.predict(image_expand, batch_size=1)
        print('start')
        print(self.master.winfo_x())
        global image
        x1 = self.master.winfo_x() + 7
        x2 = x1+256
        y1 = self.master.winfo_y() + 75
        y2 = y1+256
        frameposition = (x1,y1,x2,y2)
        image_raw = ImageGrab.grab(bbox=frameposition)
        image = np.array(image_raw, dtype=np.uint8)
        #new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        #cv2.imshow('reshaped image', new_image)
        image_expand = image[np.newaxis, :, :, :]
        print(image.shape)
        print(image_expand.shape)
        result = self.model.predict(image_expand, batch_size=1)
        print(result)
        whois = result.argmax()
        probability = result[0,whois] * 100
        result_text = f'{HOLOLIST[whois]}:{probability}, %'
        print(result_text)
        self.label_result.configure(text=result_text)




class Frame_Display(tk.Frame):
    def __init__(self, master=None):
        # Windowの初期設定を行う。
        super().__init__(master)
        self.configure(background ='snow')
        self.configure(width=256, height=256)
        self.pack(expand=1, fill=tk.BOTH)

if __name__ == "__main__":
    root = tk.Tk()
    Frame_Control(master=root)
    Frame_Display(master=root)
    root.mainloop()