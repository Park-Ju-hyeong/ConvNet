import numpy as np
import pickle
import os
import download
from dataset import one_hot_encoded

########################################################################

data_path = "data/CIFAR-100/"

data_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

########################################################################

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 100

########################################################################

def _get_file_path(filename=""):

    return os.path.join(data_path, "cifar-100-python/", filename)

def _unpickle(filename):

    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data

def _convert_images(raw):

    raw_float = (np.array(raw, dtype=float) - 127.5) / 127.5

    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):

    data = _unpickle(filename)

    raw_images = data[b'data']

    cls = np.array(data[b'fine_labels'])

    images = _convert_images(raw_images)

    return images, cls


########################################################################


def maybe_download_and_extract():

    download.maybe_download_and_extract(url=data_url, download_dir=data_path)

    
def load_class_names():

    raw = _unpickle(filename="meta")[b'fine_label_names']

    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    
    images, cls = _load_data(filename="train")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_test_data():

    images, cls = _load_data(filename="test")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################
