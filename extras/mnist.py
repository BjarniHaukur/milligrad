import os
import gzip
from tempfile import TemporaryDirectory

import requests
import numpy as np

def load_images(url:str, filename:str)->np.ndarray:
    """Download and unzip a file from a URL into memory."""
    response = requests.get(url + filename, stream=True)
    if response.status_code == 200:
        with TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            with gzip.open(temp_path, 'rb') as f_in:
                _, num_images, num_rows, num_cols = np.frombuffer(f_in.read(16), dtype=">i4", count=4, offset=0)
                images = np.frombuffer(f_in.read(), dtype=np.uint8)
                return images.reshape(num_images, num_rows, num_cols)
    else:
        raise ValueError(f"Failed to download {filename}")
    
def load_labels(url:str, filename:str)->np.ndarray:
    """Download and unzip a file from a URL into memory."""
    response = requests.get(url + filename, stream=True)
    if response.status_code == 200:
        with TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            with gzip.open(temp_path, 'rb') as f_in:
                _ = np.frombuffer(f_in.read(8), dtype=">i4", count=2, offset=0)
                return np.frombuffer(f_in.read(), dtype=np.uint8)
    else:
        raise ValueError(f"Failed to download {filename}")

def load_mnist_data():
    """Load the MNIST dataset (training and testing images and labels)."""
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    train_images = load_images(base_url, files['train_images'])
    train_labels = load_labels(base_url, files['train_labels'])
    test_images = load_images(base_url, files['test_images'])
    test_labels = load_labels(base_url, files['test_labels'])

    return (train_images, train_labels), (test_images, test_labels)

