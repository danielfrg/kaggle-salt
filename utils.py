import bcolz
from skimage.transform import resize

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
    return resize(img, (OR_IM_HEIGHT, OR_IM_WIDTH, OR_IM_CHANNEL), mode='constant', preserve_range=True, anti_aliasing=True)
    