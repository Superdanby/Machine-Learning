import pathlib
import time
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from PIL import Image

DATASET = pathlib.Path('CroppedYale').resolve()
FOLDERS = sorted([x for x in DATASET.iterdir() if x.is_dir()])
TRAIN_NUM = 35


def load_paths():
    """ Load paths """

    train_paths, train_lbls = [], []
    test_paths, test_lbls = [], []

    for i, dir_ in enumerate(tqdm(FOLDERS)):
        img_paths = dir_.glob('*.pgm')
        img_paths = [p for p in sorted(img_paths)]
        lbls = [i for _ in range(len(img_paths))]

        train_paths.extend(img_paths[:TRAIN_NUM])
        train_lbls.extend(lbls[:TRAIN_NUM])
        test_paths.extend(img_paths[TRAIN_NUM:])
        test_lbls.extend(lbls[TRAIN_NUM:])

    return train_paths, train_lbls, test_paths, test_lbls


def load_data(paths, lbls):
    """ Load data and labels."""

    # load with constant size
    # n_samples = len(paths)
    # xs = np.zeros((n_samples, 64, 64), dtype=np.float32)

    # load with dynamic size
    xs = []
    ys = np.uint8(lbls)
    for i, p in enumerate(tqdm(paths)):
        img = Image.open(p)
        xs.append(np.array(img, dtype=float))
    return np.asarray(xs), ys


# Main program starts here.
train_paths, train_lbls, test_paths, test_lbls = load_paths()
x_train, y_train = load_data(train_paths, train_lbls)
x_test, y_test = load_data(test_paths, test_lbls)

# reshape to calculate with numpy or cdist
x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

t1 = time.time()

l1_dist = cdist(x_test, x_train, 'cityblock')  # len(xtest) by len(xtrain) matrix
l1_pred = y_train[np.argmin(l1_dist, axis=1)]
# Calculate with numpy and generator, extremely slow, even for loop is faster.
# l1_pred = [y_train[np.argmin(np.sum(np.absolute(x - x_train), axis=1))] for x in x_test]

l1_acc = np.mean(l1_pred == y_test)

print(f'SAD Accuracy: {l1_acc}, {time.time() - t1:.5f}seconds')

t2 = time.time()
l2_dist = cdist(x_test, x_train, 'euclidean')  # square root of SSD
l2_pred = y_train[np.argmin(l2_dist, axis=1)]
l2_acc = np.mean(l2_pred == y_test)

print(f'SSD Accuracy: {l2_acc}, {time.time() - t2:.5f}seconds')
