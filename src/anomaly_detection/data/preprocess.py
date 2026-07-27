import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from ..config import IMAZE_SIZE


def preprocess_and_visualize(malaria_builder, size):
    dataset = malaria_builder.as_dataset(split="train")
    fig, axs = plt.subplots(size, 5, figsize=(20,20))
    axs = axs.ravel()
    # Set the titles of each image and store the images in an array
    titles = ['Original', 'Resized', 'GrayScaled', 'Normalized', 'Blurred']
    row = 0
    for image_tf in dataset.take(size):
        image = image_tf["image"]
        image_resized = tf.image.resize(image, IMAZE_SIZE).numpy()
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        image_normalized = image_gray/255.0
        image_blur = cv2.GaussianBlur(image_normalized, (5,5), 0)

        images = [image, image_resized.astype(int), image_gray, image_normalized, image_blur]

        for i, modified_image in enumerate(images):
            axs[i + row*5].imshow(modified_image, cmap='gray')
            axs[i+ row*5].set_title(titles[i])
            axs[i+ row*5].axis("off")
        row +=1
        
    plt.tight_layout()
    plt.show()

def preprocess_image_dataset(dataset):
    image = dataset['image']
    label = dataset['label']
    image_resized = tf.image.resize(image, IMAZE_SIZE)
    image_normalized = image_resized/255.0
    return {'image': image_normalized, 'label': label}






