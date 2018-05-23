import os
import numpy as np
import pandas as pd
import tensorflow as tf
import imagenet_model as model
import pathlib
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim
from skimage import io,transform
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from tqdm import tqdm
from PIL import Image

TRAINSET = pathlib.Path('train').resolve()
TRAINLBS = pathlib.Path('labels.csv').resolve()
TESTSET = pathlib.Path('test').resolve()
DEBUG = False
INPUT_SIZE = [224, 224, 3]
N_CLASSES = 120
LEARNING_RATE = 2e-5
EPOCHS = 50
BATCH_SIZE = 32
SPLIT = 1
LOAD_PRETRAIN = True
LOG_PATH = "/log/base"
VIS_CLEAR = 0
# PRETRAIN_PATH = "pretrain/model.ckpt"
PRETRAIN_PATH = "vgg_16.ckpt"


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def visual(data, title=None):
    global VIS_CLEAR
    global vis_log
    if VIS_CLEAR == 0:
        print("reset")
        vis_log = {}
        for k in data:
            vis_log[k] = []
    VIS_CLEAR = VIS_CLEAR + 1
    print(VIS_CLEAR)
    print(title)
    plt.title(title)
    plt.xlabel("epoch")
    print(data)
    for k, v in data.items():
        vis_log[k].append(v)
        print(k)
        plt.plot(range(len(vis_log[k])), vis_log[k], label = k)
        plt.legend()
        plt.savefig(str(title + ' ' + k + '.jpg'))
        plt.close()

def train_eval(sess, x_data, y_label, batch_size, train_phase, is_eval,  epoch=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    tmp_loss, tmp_acc = 0, 0
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        _, batch_loss, batch_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: x_data[start:end], y: y_label[start:end],
                                            is_training: train_phase})
        tmp_loss += batch_loss * (end - start)
        tmp_acc += batch_acc * (end - start)
    tmp_loss /= n_sample
    tmp_acc /= n_sample
    if train_phase:
        print('\nepoch: {0}, loss: {1:.4f}, acc: {2:.4f}'.format(epoch+1, tmp_loss, tmp_acc))
        visual({'Accuracy': tmp_acc, 'Loss': tmp_loss}, title='training')
    # else:
    #     visual({'Accuracy': tmp_acc, 'Loss': tmp_loss}, title='validation')

def test_eval(sess, x_data, train_phase):
    batch_size = 1
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    tmp_pred=[]
    log=[]
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        tmp_logits = sess.run(logits, feed_dict={x: x_data[start:end], is_training: train_phase})
        tmp=softmax(np.squeeze(tmp_logits))
        tmp_pred.append(tmp)
    tmp_pred = np.array(tmp_pred)

    return tmp_pred


# data preprocess by yourself
def load_paths(shuffle=False, debug=False, split=1):
    """ Load paths """

    df_train = pd.read_csv(TRAINLBS)

    paths = [TRAINSET / (p + '.jpg') for p in df_train['id']]
    lbls = [lbl for lbl in df_train['breed']]
    lbls_sorted = sorted(set(lbls))
    global N_CLASSES
    N_CLASSES = len(lbls_sorted)
    lbls_enc = {}
    lbls_dec = {}
    for i, key in enumerate(lbls_sorted):
        lbls_enc[key] = i
        lbls_dec[i] = key
    lbls = [lbls_enc[lbl] for lbl in lbls]

    if shuffle:
        zipped = list(zip(paths, lbls))
        random.shuffle(zipped)
        paths, lbls = zip(*zipped)

    # Check if labels are still linked with the right paths after shuffle
    for i, x in enumerate(paths):
        # print(x.name[:-4])
        breed = df_train[df_train['id'] == x.name[:-4]].iloc[0]['breed']
        # print(breed)
        assert breed == lbls_dec[lbls[i]]

    test_paths = sorted(list(TESTSET.glob('*.jpg')))
    # print(test_paths[:10])

    if debug:
        paths = paths[:100]
        lbls = lbls[:100]
        test_paths = test_paths[:100]

    assert split <= 1
    assert split > 0

    split_index = int(split * len(paths))
    train_paths = paths[:split_index]
    train_lbls = lbls[:split_index]
    val_paths = paths[split_index:]
    val_lbls = lbls[split_index:]

    return train_paths, train_lbls, val_paths, val_lbls, lbls_sorted, test_paths


def load_data(paths, lbls=None):
    """ Load data and labels."""

    # load with dynamic size
    xs = []
    for i, p in enumerate(tqdm(paths)):
        img = Image.open(p)
        npimg = np.asarray(img, dtype=float)
        npimg = transform.resize(npimg, (INPUT_SIZE[2], INPUT_SIZE[0], INPUT_SIZE[1]))
        # npimg = transform.resize(npimg, (INPUT_SIZE[2], INPUT_SIZE[0], INPUT_SIZE[1]), anti_aliasing=True)
        npimg = np.rollaxis(npimg, 0, 3)
        assert npimg.shape == (INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        xs.append(npimg)
    print("Data finished loading.")
    if lbls is None:
        return np.asarray(xs)
    ys = np.uint8(lbls)
    print(f'ys shape {ys.shape}')
    OHE = preprocessing.OneHotEncoder(n_values = N_CLASSES)
    ys = OHE.fit_transform(ys.reshape(-1,1)).toarray()
    print(ys.shape)
    return np.asarray(xs), ys


if __name__ == '__main__':

    train_paths, train_lbls, val_paths, val_lbls, lbls_sorted, test_paths = load_paths(shuffle=False, debug=DEBUG, split=SPLIT)
    train_data, train_label = load_data(train_paths, train_lbls)
    if SPLIT != 1:
        val_data, val_label = load_data(val_paths, val_lbls)

    test_data = load_data(test_paths)
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)

    x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, N_CLASSES), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='train_phase')

    # tf.reset_default_graph()
    logits = model.VGG16(inputs=x, is_training=is_training, n_classes=N_CLASSES)

    with tf.name_scope('LossLayer'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1)), tf.float32))

    init = tf.global_variables_initializer()

    # What's this?
    # restore_variable = [var for var in tf.global_variables() if var.name.startswith('')]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if LOAD_PRETRAIN:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            restore = slim.assign_from_checkpoint_fn(model_path = PRETRAIN_PATH, var_list = slim.get_model_variables('vgg_16'))
            # saver.restore(sess, PRETRAIN_PATH)
            sess.run(init_op)
            restore(sess)
        else:
            sess.run(init)

        for i in range(EPOCHS):
            train_eval(sess=sess, x_data=train_data, y_label=train_label, batch_size=BATCH_SIZE,
                    train_phase=True, is_eval=False, epoch=i)
            # if SPLIT != 1:
            #     train_eval(sess=sess, x_data=val_data, y_label=val_label, batch_size=BATCH_SIZE,
            #                 train_phase=False, is_eval=False, epoch=i)
        saver = tf.train.Saver(slim.get_model_variables())
        saver.save(sess, 'model/model.ckpt')
        ans = test_eval(sess=sess, x_data=test_data, train_phase=False)

    # output
    opt = pd.DataFrame(data = ans, columns=lbls_sorted)
    ids = [fpath.name[:-4] for fpath in test_paths]
    opt.insert(loc = 0, column = 'id', value=ids)
    opt.to_csv('prediction.csv', sep = ',', index = False)
