import bcolz

OR_IM_WIDTH = 101
OR_IM_HEIGHT = 101
OR_IM_CHANNEL = 101

IM_WIDTH = 128
IM_HEIGHT = 128
IM_CHAN = 1


def save_arr (fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    

def load_array(fname):
    return bcolz.open(fname)[:]