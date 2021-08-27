import math
import os
import sys

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def model():
    SRCNN = tf.keras.Sequential(name="SRCNN")
    SRCNN.add(
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(9, 9),
            padding="VALID",
            use_bias=True,
            input_shape=(None, None, 1),
            kernel_initializer="glorot_uniform",
            activation="relu",
        )
    )
    SRCNN.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            activation="relu",
        )
    )
    SRCNN.add(
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(5, 5),
            padding="VALID",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            activation="linear",
        )
    )
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

    # Compile model
    SRCNN.compile(
        optimizer=optimizer, loss="mean_squared_error", metrics=["mean_squared_error"]
    )

    return SRCNN


def SRCNN_Predict(target):
    # Normalize
    Y = np.zeros((1, target.shape[0], target.shape[1], 1), dtype=np.float32)
    Y[0, :, :, 0] = target[:, :, 0].astype(np.float32) / 255.0
    # Predict
    srcnn_model = model()
    srcnn_model.load_weights("work_dirs/SRCNN/srcnn")
    pre = srcnn_model.predict(Y, batch_size=1) * 255.0

    # Post process output
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)

    # Copy y channel back to image and convert to BGR
    output = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
    output[6:-6, 6:-6, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
    # Save image
    cv2.imwrite("./SRCNN_Result/" + filename, output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return output
