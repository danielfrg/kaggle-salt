import math
import random 

import tqdm
import bcolz
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image as image_utils


OR_IM_WIDTH = 101
OR_IM_HEIGHT = 101
OR_IM_CHANNEL = 3

IM_WIDTH = 128
IM_HEIGHT = 128
IM_CHAN = 1


def save_arr (fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    

def load_array(fname):
    return bcolz.open(fname)[:]


def upsample(img):
    return resize(img, (IM_HEIGHT, IM_WIDTH, IM_CHAN), mode='constant', preserve_range=True, anti_aliasing=True)

    
def downsample(img):
    return resize(img, (OR_IM_HEIGHT, OR_IM_WIDTH), mode='constant', preserve_range=True, anti_aliasing=True)


def rle_decode(rle, shape):
    """
    rle: run-length string or list of pairs of (start, length)
    shape: (height, width) of array to return 
    Returns
    -------
        np.array: 1 - mask, 0 - background
    """
    if isinstance(rle, float) and math.isnan(rle):
        rle = []
    if isinstance(rle, str):
        rle = [int(num) for num in rle.split(' ')]
    # [0::2] means skip 2 since 0 until the end - list[start:end:skip]
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    img = img.reshape(1, shape[0], shape[1])
    img = img.T
    return img


def rle_encode(img):
    """
    img: np.array: 1 - mask, 0 - background
    Returns
    -------
    run-length string of pairs of (start, length)
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle if rle else float('nan')


def iou(y_true, y_pred):
    """ Intersection over Union Metric
    """
    component1 = y_true.astype(dtype=bool)
    component2 = y_pred.astype(dtype=bool)

    overlap = component1 * component2 # Logical AND
    union = component1 + component2 # Logical OR

    iou = overlap.sum() / float(union.sum())
    return iou


def iou_batch(y_true, y_pred):
    batch_size = y_true.shape[0]
    metric = []
    for i in range(batch_size):
        value = iou_metric(y_true[i], y_pred[i])
        metric.append(value)
    return np.mean(metric)


def mean_iou(y_true, y_pred):
    """Keras valid metric to use with a keras.Model
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def plot_arrays(*args, n=6, figsize=(15, 5), randomize=True, strata=None):
    """
    args: single or multiple arrays to plot (n, height, width, channel)
    n: number of images to plot from the array
    strata: if not None it will plot one image per stratum
    """
    fig, axarr = plt.subplots(len(args), n, figsize=figsize)
    for i in range(n):
        arrays = args
        if strata is not None:
            arrays = []
            for arr in args:
                arrays.append(arr[strata == i])
        if randomize:
            rand = random.randint(0, arrays[0].shape[0])
        for y, array in enumerate(arrays):
            image = array[rand]
            if image.shape[2] == 1:
                # This is if we reduced it to only one channel
                image = image_utils.array_to_img(array[rand])
            if len(args) == 1:
                axarr[i].imshow(image)
            else:
                axarr[y, i].imshow(image)
    fig.tight_layout()
    

def gen_strata(target, n=11):
    coverage_pct = np.zeros((target.shape[0], ), dtype=np.float64)
    
    for i, mask in tqdm.tqdm_notebook(enumerate(target), total=target.shape[0]):
        coverage_pct[i] = np.mean(mask) / 255
        
    def cov_to_class(val):    
        for i in range(0, n):
            if val * (n - 1) <= i :
                return i
    v_cov_to_class = np.vectorize(cov_to_class)
    strata = v_cov_to_class(coverage_pct)
        
    return coverage_pct, strata