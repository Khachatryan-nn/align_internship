import os
import imgaug
import keras_ocr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

def load_single_image(image_path):
    image = Image.open(image_path)
    image_tf = keras.preprocessing.image.img_to_array(image)
    return image_tf

def extract_id(image_file):
    print(image_file)
    image = load_single_image(image_file)
    #pipeline = keras_ocr.pipeline.Pipeline(
        
    #)

    #result = pipeline.recognize([image])
    result = np.random.randint(low=0, high=100, size=(1, 1, 1))
    return result[0][0][0]