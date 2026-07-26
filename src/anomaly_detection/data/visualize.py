import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_by_dataset(datasets):
    fig , axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.ravel()
    for index, cell_image_data in enumerate(datasets):
        cell_image = cell_image_data['image']
        cell_image_label = cell_image_data['label']
        #resize to 100 * 100
        cell_image = tf.image.resize(cell_image, [100, 100])
        axs[index].imshow(cell_image.numpy().astype("uint8"))
        axs[index].title.set_text(f'{"Parasitized" if cell_image_label.numpy() else "Uninfected"}')
        axs[index].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize(malaria_builder, batch_size=10):
    malaria_dataset = malaria_builder.as_dataset(split = "train")
    visualize_by_dataset(malaria_dataset.take(batch_size))