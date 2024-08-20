import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set up the root path and classes
ROOT = r'D:\dev\data'
CLASS_NAME = sorted(list(os.listdir(os.path.join(ROOT, 'train'))))

# Function to read and process images
def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)

def folder_to_images(folder, size):
    list_dir = [os.path.join(folder, name) for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path

# Function to calculate L1 distance
def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)

# Function to get L1 score
def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            rates = absolute_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

# Function to plot results
def plot_results(query_path, ls_path_score, reverse=False):
    ls_path_score.sort(key=lambda x: x[1], reverse=reverse)
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    for i, (img_path, score) in enumerate(ls_path_score[:5]):
        img = Image.open(img_path)
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'Score: {score:.2f}')
    plt.show()

if __name__ == "__main__":
    root_img_path = os.path.join(ROOT, 'train')
    query_path = os.path.join(ROOT, r'test\bullet_train\n02917067_5772.JPEG')
    size = (448, 448)
    query, ls_path_score = get_l1_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

