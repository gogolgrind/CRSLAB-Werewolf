import fnmatch
import numpy as np
import scipy as sp
import os 

def qsplit(im):
    h,w,_ = im.shape
    q = [
        im[0:h//2,0:w//2],
        im[0:h//2,w//2:w],
        im[h//2:h,0:w//2],
        im[h//2:h,w//2:w],
    ]
    return q

def file_list(path,pat):
    """
        return list with files names relevant to mask
        Parameters:
        ===========
        path : str, path to folder with fikes
        pat : str, mask e.g *.mp4
    """
    lst = fnmatch.filter(os.listdir(path), pat)
    return np.sort(np.array([os.path.join(path,e) for e in lst]))

def t2sec(dt):
    t = dt.strftime("%H:%M:%S")
    return sum(int(i) * 60**index for index, i in enumerate(t.split(":")[::-1]))

def sliding_window(sequence,win_size,step=1):
    """
        Returns a generator that will iterate through
        the defined chunks of input sequence.  Input sequence
        must be iterable.
        src link : https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
    """
    # Verify the inputs
    try: 
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(win_size) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > win_size:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if win_size > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    nb_chunks = ((len(sequence)-win_size)//step)+1
    # Do the work
    for i in range(0,nb_chunks*step,step):
        yield sequence[i:i+win_size]
        
def balanced_sample(X, y, sample_size, random_state = None):
    """ 
        return a balanced data set by sampling all classes with sample_size 
        current version is developed on assumption that the positive
        class is the minority.
        Parameters:
        ===========
        X: {numpy.ndarrray}
        y: {numpy.ndarray}

        original src:
            http://stackoverflow.com/questions/23455728/scikit-learn-balanced-subsampling
    """
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_state is None:
        np.random.seed(random_state)

    #  find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of positive label
    sample_size = uniq_counts[0]
    over_sample_idx = np.random.choice(groupby_levels[1], size=sample_size, replace=True).tolist()
    balanced_copy_idx = groupby_levels[0] + over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx, :], y[balanced_copy_idx]

def get_class_weight(labels_dict,mu = 0.5):
    """
        src : http://datascience.stackexchange.com/a/16467
    """
    total = np.sum([e for e in labels_dict.values() ])
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = np.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight