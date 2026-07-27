from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ..config import IMAZE_SIZE
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def augment_image(data):
    label = data['label']
    image = data['image']
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return {'image': image, 'label': label}

def augment(datasets, batch_size=0, visualize=False):
    # datasets  : the data splits (train/val/test) to augment
    # batch_size : how many images to process at a time
    # visualize  : if True, use ImageDataGenerator to preview augmentations on an image
    if (visualize):
        batch_size = 1
        # use ImageDataGenerator
        # Configure random transformations for image augmentation
        data_gen = ImageDataGenerator(              # the "augmentation machine": makes random variations of an image
            rotation_range=90,                      # rotate randomly up to +/- 90 degrees
            width_shift_range=0.1,                  # slide left/right by up to 10% of the width
            height_shift_range=0.1,                 # slide up/down by up to 10% of the height
            shear_range=0.1,                        # slant/skew the image (push top & bottom opposite ways)
            zoom_range=0.1,                         # zoom in/out randomly by up to 10%
            horizontal_flip=True,                   # randomly mirror the image left-to-right
            vertical_flip=True,                     # randomly mirror the image top-to-bottom
        )
        single_image = next(iter(datasets.take(1)))['image']
        single_image = tf.image.resize(single_image, IMAZE_SIZE)

       # aug_iter = data_gen.flow([single_image]) //causing exception

        single_image = np.expand_dims(single_image, axis = 0)
        aug_iter = data_gen.flow(single_image)

        fig , axs = plt.subplots(3, 3, figsize=(10, 10))
        axs = axs.ravel()

        for axis in axs:
            augmented_image = next(aug_iter)[0].astype('uint8')
            axis.imshow(augmented_image)
            axis.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        # use tensorflow api
        if(batch_size == 0):
            batch_size = len(datasets)
        return datasets.take(batch_size).map(augment_image)
        

        
