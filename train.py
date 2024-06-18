#import os
#import imgaug
#import keras_ocr
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.losses import SparseCategoricalCrossentropy
#import numpy as np
#from sklearn.model_selection import train_test_split
#from PIL import Image
#import cv2

#pipeline = keras_ocr.recognition.Recognizer()

## Load the image with specified path
#image_path = './train/4-2.JPG'
#def load_single_image(image_path):
#    image = Image.open(image_path)
#    image_tf = keras.preprocessing.image.img_to_array(image)
#    return image_tf
    
#def load_data(folder):
#    data = []
    
#    for filename in os.listdir(folder):
#        image_label, _ = filename.split('.')
#        image_path = os.path.join(folder, filename)
#        image = Image.open(image_path)
#        image_tf = keras.preprocessing.image.img_to_array(image)
#        data.append([image_tf, image_label])
        
#    return data

#def prepare_data(train_data, test_data, batch_size=32):
#    augmenter = imgaug.augmenters.Sequential([
#        imgaug.augmenters.Fliplr(0.5),
#    ])
#    #train_dataset = 

#def train(model,
#          train_data,
#          test_data,
#          optimizer,
#          loss_fn,
#          num_epochs=10
#          ):
#    for epoch in range(num_epochs):
#        print(f"Epoch {epoch+1}/{num_epochs}")
#        for image, label in train_data:
#            # Preprocess the image and label
#            image = keras_ocr.tools.read(image)
#            label = model.detector.prepare_labels([label])

#            with tf.GradientTape() as tape:
#                # Forward pass
#                prediction = model.recognizer.model(image, training=True)

#                # Compute the loss
#                loss = loss_fn(label, prediction)

#            # Optimize
#            gradients = tape.gradient(loss, model.recognizer.model.trainable_variables)
#            optimizer.apply_gradients(zip(gradients, model.recognizer.model.trainable_variables))

#        print(f"Loss: {loss.numpy()}")

#    # Save the fine-tuned model
#    #model.recognizer.model.save('fine_tuned_ocr_model')

##if __name__ == '__main__':
##    data = load_data('./train')
##    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
##    train_dataset, test_dataset = prepare_data(train_data, test_data)
##    model = keras_ocr.recognition.Recognizer()
##    model.compile()
##    optimizer = Adam(learning_rate=1e-3)
##    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
##    train(model, train_data, test_data, optimizer, loss_fn)