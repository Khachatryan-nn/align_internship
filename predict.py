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

def extract_id(image_path):
    image_path = os.path.join(os.getcwd(), image_path)
    image = load_single_image(image_path)
    #pipeline = keras_ocr.pipeline.Pipeline()

    #result = pipeline.recognize([image])
    result = [[[0]]]
    return result[0][0][0]