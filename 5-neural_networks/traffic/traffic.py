import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Looping over folders in data directory
    for category in os.listdir(data_dir):

        # Assembling folder path, then looping over files in folder path
        folder_path = os.path.join(data_dir, category)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            # Loads files into numpy nd.array, and resizing to the 30x30 specification
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

                # Adding image and its category into lists
                images.append(image)
                labels.append(category)

    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer of 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        # Average pooling layer using a 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        
        # When using a pool size of 2, we effectively downsample the 
        # image by a factor of 2, so in the following Convolutional layer
        # we would want to multiply the number of channels by 2

        # Convolutional layer of 64 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # Average pooling layer using a 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        
        # Convolutional layer of 128 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # Average pooling layer using a 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
