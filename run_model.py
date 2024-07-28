#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:06:58 2024

@author: hmoqadam
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from models.unet_deeper import create_model  # Adjust this import based on your actual model creation function
from utils.data_loader import load_data
from utils.metrics import focal_tversky_loss, binary_accuracy

# Define Focal Tversky Loss
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = numerator + alpha * tf.reduce_sum((1 - y_true) * y_pred) + beta * tf.reduce_sum(y_true * (1 - y_pred))
    tversky = numerator / denominator
    return tf.pow(1 - tversky, gamma)

# Define Leaky ReLU activation function
def leaky_relu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha=alpha)

# Load data
X_train, Y_train, X_test, Y_test = load_data()

# Define the model
model = create_model(input_shape=(512, 512, 1), activation_function='leaky_relu', dropout_rate=0.3, l2_lambda=0.001)

# Compile the model
optimizer = Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss=focal_tversky_loss, metrics=[binary_accuracy])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=64)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Save the model
model.save('final_unet_deep_model.h5')

# Plot training accuracy and loss
plt.plot(history.history['binary_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()



