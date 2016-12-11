import numpy
from collections import OrderedDict
import cPickle
import gzip
import os
import cv
import cv2
import lmdb
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
from tensorflow.python.platform import gfile
import thread_utils

class ReplayMemory:
    def __init__(self, batch_size, buffer_size, folder,
                 num_threads=1):
        self.folder = folder
        self.buffer_size = buffer_size
        gfile.MakeDirs(self.folder)
        self.fnames = []
        self.batch_size = batch_size
        self.batch_fetcher = thread_utils.ThreadedRepeater(self._get_tup, num_threads, batch_size)

    def add_transition(self, tup):
        if len(self.fnames) < self.buffer_size:
            fname ="{}/{}.npz".format(self.folder, random.random())
            self.fnames.append(fname)
        else:
            fname = random.choice(self.fnames)
        with open(fname, 'w') as f:
            np.save(f, tup)

    def _get_tup(self):
        try:
            if len(self.fnames) < self.batch_size:
                return None
            fname = random.choice(self.fnames)
            with open(fname, 'r') as f:
                tup = np.load(f)
            return [tup]
        except:
            print("Failed to Load")
            return self._get_tup()

    def get_batch(self):
        batch = self.batch_fetcher.run()
        return zip(*batch)

class SequenceReplayMemory:
    def __init__(self, batch_size, buffer_size, folder, num_threads=1, max_length=10, random_start=True):
        self.folder = folder
        self.buffer_size = buffer_size
        gfile.MakeDirs(self.folder)
        self.fnames = []
        self.batch_size = batch_size
        self.max_length = max_length
        self.random_start = random_start
        self.batch_fetcher = thread_utils.ThreadedRepeater(self._get_seq, num_threads, batch_size)

    def add_sequence(self, sequence):
        """
        sequence is a list of tuples in form (observation, action, reward, new_obs, done)
        """
        if len(self.fnames) < self.buffer_size:
            fname = "{}/{}.npz".format(self.folder, random.random())
            self.fnames.append(fname)
        else:
            fname = random.choice(self.fnames)
        with open(fname, 'w') as f:
            np.save(f, sequence)

    def _get_seq(self):
        try:
            if len(self.fnames) < self.batch_size:
                return None
            fname = random.choice(self.fnames)
            with open(fname, 'r') as f:
                seq = np.load(f)
                if len(seq) < self.max_length:
                    # get number of frames to zero pad
                    diff = self.max_length - len(seq)
                    obs, acts, rs, new_obs, dones = zip(*seq)
                    obs = np.array(obs)
                    acts = np.array(acts)
                    rs = np.array(rs)
                    new_obs = np.array(new_obs)
                    dones = np.array(dones)

                    obs = np.concatenate([obs, np.zeros([diff] + list(obs.shape[1:]))])
                    acts = np.concatenate([acts, np.zeros([diff] + list(acts.shape[1:]))])
                    rs = np.concatenate([rs, np.zeros([diff] + list(rs.shape[1:]))])
                    new_obs = np.concatenate([new_obs, np.zeros([diff] + list(new_obs.shape[1:]))])
                    dones = np.concatenate([dones, np.zeros([diff] + list(dones.shape[1:]))])

                    return [(obs, acts, rs, new_obs, dones)]
                else:
                    # choose a start frame
                    start_frame = random.randint(0, len(seq) - self.max_length)
                    chosen_frames = seq[start_frame:start_frame+self.max_length]
                    obs, acts, rs, new_obs, dones = zip(*chosen_frames)
                    return [(np.array(obs), np.array(acts), np.array(rs),
                            np.array(new_obs), np.array(dones))]

        except:
           print("Failed to load")
           return self._get_seq()
    def get_batch(self):
        batch = self.batch_fetcher.run()
        # batch is list of (obs, acts, rs, new_obs, dones)
        # zip to get batch
        return zip(*batch)

class ActionSequenceReplayMemory:
    def __init__(self, batch_size, buffer_size, folder, num_threads=1, max_length=10):
        self.folder = folder
        self.buffer_size = buffer_size
        gfile.MakeDirs(self.folder)
        self.fnames = []
        self.batch_size = batch_size
        self.max_length = max_length
        self.batch_fetcher = thread_utils.ThreadedRepeater(self._get_seq, num_threads, batch_size)

    def add_sequence(self, sequence):
        """
        sequence is a list of tuples in form (observation, actions (list), final observation)
        """
        if len(self.fnames) < self.buffer_size:
            fname = "{}/{}.npz".format(self.folder, random.random())
            self.fnames.append(fname)
        else:
            fname = random.choice(self.fnames)
        with open(fname, 'w') as f:
            np.save(f, sequence)

    def _get_seq(self):
        try:
            if len(self.fnames) < self.batch_size:
                return None
            fname = random.choice(self.fnames)
            with open(fname, 'r') as f:
                (obs, actions, new_obs) = np.load(f)
                # get number of frames to zero pad
                n_actions = len(actions)
                assert n_actions <= self.max_length
                z_pad = [0] * (self.max_length - n_actions)
                padded_actions = actions + z_pad

                return [(obs, np.array(padded_actions), n_actions, new_obs)]

        except:
           #print("Failed to load")
           return self._get_seq()
    def get_batch(self):
        batch = self.batch_fetcher.run()
        # batch is list of (obs, acts, lens, new_obs)
        # zip to get batch
        return zip(*batch)



def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def generate_train_test_validation_sets(x,y,train_percent,validation_percent):

    n_samples = numpy.asarray(x).shape[1]

    permuted_indices = numpy.random.permutation(n_samples)
    print("permutation")
    print(permuted_indices)

    train_int = (1-train_percent-validation_percent)*n_samples

    test_int  = (1-validation_percent)

    train = [[],[]]
    test  = [[],[]]
    valid = [[],[]]

    for i in permuted_indices[0:train_int]:

        train[0].append(x[:][:][i])
        train[1].append(y[:][:][i])

    for i in permuted_indices[train_int:test_int]:

        test[0].append(x[:][:][i])
        test[1].append(y[:][:][i])

    for i in permuted_indices[test_int:]:

        valid[0].append(x[:][:][i])
        valid[1].append(y[:][:][i])

    return train, test, valid

def load_svhn_data():

    d1 = './SVHN/train'
    d2 = './SVHN/extra'
    d3 = './SVHN/test'

    training_set_size = 33403+202350
    train = numpy.zeros((training_set_size,3,100,45))
    for i in range(1,33403,1):
        directory = d1+'/'+str(i)+'.png'
        #print directory
        image = cv2.imread(directory)
        try:
            dst = cv2.resize(image, (100, 45), interpolation = cv2.INTER_CUBIC)
        except:
            print directory
        dst = numpy.swapaxes(dst,0,2)

        train[i,] = dst

    extra_index = 1
    for i in range(33403,training_set_size,1):
        directory = d2+'/'+str(extra_index)+'.png'
        #print directory
        image = cv2.imread(directory)
        try:
            dst = cv2.resize(image, (100, 45), interpolation = cv2.INTER_CUBIC)
        except:
            print directory
        dst = numpy.swapaxes(dst,0,2)

        train[i,] = dst

        extra_index += 1

    idx_list = numpy.arange(training_set_size)
    numpy.random.shuffle(idx_list)
    valid = train[idx_list[0:40000],]
    train = train[idx_list[40000:]]

    #Load the testing data
    test = numpy.zeros((13069,3,100,45))
    for i in range(1,13069,1):
        directory = d3+'/'+str(i)+'.png'
        image = cv2.imread(directory)
        try:
            dst = cv2.resize(image, (100, 45), interpolation = cv2.INTER_CUBIC)
        except:
            print directory
        dst = numpy.swapaxes(dst,0,2)

        test[i,] = dst

    width  = 100
    height = 45

    return [(train,[]), (valid,[]), (test,[]), width, height, 3]

def load_cifar_data():

    d1 = './data/cifar-10-batches-py/data_batch_1'
    d2 = './data/cifar-10-batches-py/data_batch_2'
    d3 = './data/cifar-10-batches-py/data_batch_3'
    d4 = './data/cifar-10-batches-py/data_batch_4'
    d5 = './data/cifar-10-batches-py/data_batch_5'
    d6 = './data/cifar-10-batches-py/test_batch'
    print '... loading data'

    # Load the dataset
    f1 = open(d1, 'rb')
    train_set_1 = cPickle.load(f1)
    f1.close()
    f2 = open(d2, 'rb')
    train_set_2 = cPickle.load(f2)
    f2.close()
    f3 = open(d3, 'rb')
    train_set_3 = cPickle.load(f3)
    f3.close()
    f4 = open(d4, 'rb')
    train_set_4 = cPickle.load(f4)
    f4.close()
    f5 = open(d5, 'rb')
    train_set_5 = cPickle.load(f5)
    f5.close()

    f_train = open(d6, 'rb')
    test_set = cPickle.load(f_train)
    f_train.close()

    train_set_x = numpy.vstack((train_set_1['data'],train_set_2['data']))
    train_set_x = numpy.vstack((train_set_x,train_set_3['data']))
    train_set_x = numpy.vstack((train_set_x,train_set_4['data']))
    train_set_x = numpy.vstack((train_set_x,train_set_5['data']))
    train_set_x = train_set_x.reshape((-1,3,32,32)).transpose([0,2,3,1])
    train_set_x = numpy.asarray((train_set_x)/255., dtype='float32')

    train_set_y = train_set_1['labels']
    train_set_y = train_set_y + train_set_2['labels']
    train_set_y = train_set_y + train_set_3['labels']
    train_set_y = train_set_y + train_set_4['labels']
    train_set_y = train_set_y + train_set_5['labels']
    train_set_y = numpy.asarray(train_set_y)

    print "min", np.max(train_set_x)
    print "max", np.min(train_set_x)
    idx_list = numpy.arange(len(train_set_y))
    numpy.random.shuffle(idx_list)

    valid_set_x = train_set_x[idx_list[0:2000],:,:,:]
    valid_set_y = train_set_y[idx_list[0:2000]]
    train_set_x = train_set_x[idx_list[2000:],:,:,:]
    train_set_y = train_set_y[idx_list[2000:]]

    test_set_x = test_set['data'].reshape((-1,3,32,32)).transpose([0,2,3,1])

    test_set_x = (test_set_x)/255.
    test_set_y = numpy.asarray(test_set['labels'])

    print len(train_set_y), len(valid_set_y), len(test_set_y)


    width = 32
    height = 32

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), width, height, 3, 10]

def load_cluttered_mnist_data():
    DIM = 60
    data = numpy.load('./MNIST_Cluttered/mnist_cluttered_60x60_6distortions.npz')
    X_train, y_train = data['x_train'], numpy.argmax(data['y_train'], axis=-1)
    X_valid, y_valid = data['x_valid'], numpy.argmax(data['y_valid'], axis=-1)
    X_test, y_test = data['x_test'], numpy.argmax(data['y_test'], axis=-1)

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], DIM, DIM))
    X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM))
    X_test = X_test.reshape((X_test.shape[0], DIM, DIM))

    print "Train samples:", X_train.shape
    print "Validation samples:", X_valid.shape
    print "Test samples:", X_test.shape

    num_labels = 10
    #[(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), width, height, num_labels]
    return [(X_train, y_train), (X_valid, y_valid), (X_test, y_test), DIM, DIM, num_labels]



def load_mnist_data(mode="2D"):
    assert mode in ("2D", "flat")
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    dataset = "./data/mnist.pkl.gz"


    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)


    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.


    print 'max'
    print test_set[0].max()
    print 'min'
    print test_set[0].min()

    width = 28
    height = 28
    channels = 1
    num_labels = 10
    if mode == "flat":
        return [train_set, valid_set, test_set, width, height, channels, num_labels]
    elif mode == "2D":
        trd, trl = train_set
        vad, val = valid_set
        ted, tel = test_set
        train_data = trd.reshape((-1, height, width, channels))
        valid_data = vad.reshape((-1, height, width, channels))
        test_data  = ted.reshape((-1, height, width, channels))
        return [(train_data, trl), (valid_data, val), (test_data, tel), width, height, channels, num_labels]
    else:
        assert False



def unrar_image_net_tarball():

    dataset_path = '/data/fall11_whole'
    import os
    ld = os.listdir(dataset_path)
    #print ld
    count = 0

    for file in os.listdir(dataset_path):
        if file.endswith('.tar'):
            count += 1
            os.system('tar -xf '+dataset_path+'/'+file+' '+'-C '+dataset_path)
            os.system('rm '+dataset_path+'/'+file)
            print file

    print count

def load_VOC_2012():

    import VOC_Utils
    reload(VOC_Utils)

    train, train_labels = VOC_Utils.load_data_multilabel('train')

    valid, valid_labels = VOC_Utils.load_data_multilabel('val')

    return (train, train_labels), (valid, valid_labels), ([],[]), 3, 256, 256

def get_voc_image_batch(dirs):

    return 0

def load_imagenet(dataset_path):

    ld = os.listdir(dataset_path)

    train = []
    fc = 0
    for dir in ld:
        if os.path.isdir(dataset_path+'/'+dir):
            for file in os.listdir(dataset_path+'/'+dir):
                if file.endswith('.JPEG'):
                    train.append(dataset_path+'/'+dir+'/'+file)
                    fc +=1
                    #print dataset_path+'/'+dir+'/'+file
                #os.system("MP4Box -hint "+file)
        #print fc

    train = numpy.asarray(train)
    idx_list = numpy.arange(len(train))
    numpy.random.shuffle(idx_list)
    valid = train[idx_list[0:30000]]
    test = train[idx_list[30000:60000]]
    train = train[idx_list[60000:]]


    return (train, []), (valid, []), (test, []), 3, 256, 256

def get_imagenet_image_batch(dirs):

    out = numpy.zeros((len(dirs),3,256,256),dtype='float32')
    num = 0
    for dir in dirs:

        im = cv2.imread(dir)
        im_n = numpy.swapaxes(im,0,2)

        out[num,] = im_n

        num = num + 1

    out[:,0,:,:] = out[:,0,:,:] - 103.939
    out[:,1,:,:] = out[:,1,:,:] - 116.779
    out[:,2,:,:] = out[:,2,:,:] - 123.68

    return out

def load_sports_1m():

    dataset_path = '/data/GoogleSports'
    import os
    ld = os.listdir(dataset_path)
    #print ld
    train = numpy.asarray([])
    for dir in ld:
        if os.path.isdir(dataset_path+'/'+dir):
            for file in os.listdir(dataset_path+'/'+dir):
                if file.endswith('.mp4'):
                    train = numpy.append(train,dataset_path+'/'+dir+'/'+file)
                #os.system("MP4Box -hint "+file)

    idx_list = numpy.arange(len(train))
    numpy.random.shuffle(idx_list)
    valid = train[idx_list[0:1000]]
    test = train[idx_list[1000:2000]]
    print len(train)

    return (train, []), (valid, []), (test, []), 3, 240, 320

def load_ucf_101(dataset_path, color=False):

    #Output a video array with shape (num_frames, num_channels, height, width)

    full_path = dataset_path + "/ucfTrainTestlist/classInd.txt"
    class_list = numpy.loadtxt(full_path,dtype={'names':('classnum','classname'),'formats':('i4','S20')})

    class_dict = {}
    for tuple in class_list:
        class_dict[tuple[1]] = tuple[0]

    #print class_dict

    full_path = dataset_path + "/ucfTrainTestlist/trainlist01.txt"
    train_split_01 = numpy.loadtxt(full_path, dtype={'names':('videopath','classnum'),'formats':('S100','i4')})
    # full_path = dataset_path + "/ucfTrainTestlist/trainlist02.txt"
    # train_split_02 = numpy.loadtxt(full_path, dtype={'names':('videopath','classnum'),'formats':('S100','i4')})
    # full_path = dataset_path + "/ucfTrainTestlist/trainlist03.txt"
    # train_split_03 = numpy.loadtxt(full_path, dtype={'names':('videopath','classnum'),'formats':('S100','i4')})

    # train_split = numpy.concatenate((train_split_01,numpy.concatenate((train_split_02,train_split_03))))
    train_split = train_split_01

    full_train_split = numpy.asarray([])
    full_train_label_split = numpy.asarray([])
    for tuple in train_split:

        full_train_split = numpy.append(full_train_split,dataset_path+"/UCF-101/"+tuple[0])
        full_train_label_split = numpy.append(full_train_label_split, tuple[1])

    full_path = dataset_path + "/ucfTrainTestlist/testlist01.txt"
    test_split_01 = numpy.loadtxt(full_path, dtype='S100')
    # full_path = dataset_path + "/ucfTrainTestlist/testlist02.txt"
    # test_split_02 = numpy.loadtxt(full_path, dtype='S100')
    # full_path = dataset_path + "/ucfTrainTestlist/testlist03.txt"
    # test_split_03 = numpy.loadtxt(full_path, dtype='S100')

    # test_split = numpy.concatenate((test_split_01,numpy.concatenate((test_split_02,test_split_03))))
    test_split = test_split_01

    full_test_split = numpy.asarray([])
    full_test_split_labels = numpy.asarray([])
    for dir in test_split:

        activity = dir.split('/')[0]
        full_test_split = numpy.append(full_test_split, dataset_path+"/UCF-101/"+dir)
        full_test_split_labels = numpy.append(full_test_split_labels, class_dict[activity])

    #print full_test_split_labels
    #print full_test_split_labels.shape

    idx_list = numpy.arange(len(train_split))
    numpy.random.shuffle(idx_list)
    #print full_train_split[0]
    full_valid_split = full_train_split[idx_list[0:100]]
    full_valid_split_labels = full_train_label_split[idx_list[0:100]]

    full_train_split = full_train_split[idx_list[100:]]
    full_train_label_split = full_train_label_split[idx_list[100:]]

    if color:
        channels = 3
    else:
        channels = 1

    print full_train_label_split.shape

    return (full_train_split, full_train_label_split), (full_valid_split, full_valid_split_labels), (full_test_split, full_test_split_labels), channels, 240, 320

#Return a tensor of video data.
def get_ucf_video_set(train_test, vid_num_list, dbname):

    video_list = []
    for i in vid_num_list:

        vid, c = get_ucf_video(train_test, i, dbname)
        video_list.append((vid,c))

    return video_list

def get_ucf_video(prefix, vid_num, dbname):

    env = lmdb.open(dbname, readonly=True)
    str_id = '{:08}'.format(vid_num)
    str_id = prefix+str_id
    print str_id
    txn = env.begin()
    raw_datum = txn.get(str_id)

    return raw_datum == None

from multiprocessing.pool import ThreadPool
class threaded_video_processor():

    def __init__(self, num_processes):

        self.pool = ThreadPool(processes=num_processes)

    def threaded_video_directory_access(self, index, full_video_list, full_label_list, vid_shape, mask_shape, random_vid, get_random, get_label, step_size=1):

        result = self.pool.apply_async(directory_to_numpy, (index, full_video_list, full_label_list, vid_shape, mask_shape, random_vid, get_random, get_label, step_size) )
        return result

def get_random_frame(video_list, total_vid_num):

    vid_index       = numpy.random.randint(0,total_vid_num-1)
    vid_name        = video_list[vid_index]
    cap             = cv.CaptureFromFile(vid_name)
    vid_len          = int(cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_COUNT))
    random_vid_frame_index = numpy.random.randint(0,vid_len-1)
    cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_POS_FRAMES,random_vid_frame_index)
    img               = cv.QueryFrame(cap)
    store             = numpy.asarray(cv.GetMat(img))
    store             = numpy.swapaxes(store,0,2)

    return store, vid_name


def directory_to_numpy(index, full_video_list, full_label_list, vid_shape, mask_shape, random_vid_sampling, random_frame_sampling, get_label, step_size=1):

    vid_num = 0

    num_frames = vid_shape[0]
    batch_size = vid_shape[1]
    color   = vid_shape[2]
    width   = vid_shape[3]
    height  = vid_shape[4]

    vid_tensor_container      = numpy.zeros(vid_shape,  dtype='float32')
    rand_vid_tensor_container = numpy.zeros(vid_shape,  dtype='float32')
    mask_tensor_container     = numpy.zeros(mask_shape, dtype='float32')
    labels_tensor_container   = numpy.zeros(batch_size, dtype='int64')

    total_vid_num = full_video_list.__len__()
    #print len
    while vid_num < batch_size:
        #Sample a video from the full video list.
        if random_vid_sampling:
            video_index = numpy.random.randint(0,total_vid_num)
        else:
            video_index = index[vid_num]

        dir = full_video_list[video_index]

        #Insert the label.  If no label is available insert a dummy value.
        if get_label:
            label = full_label_list[video_index]


        #Attempt to load the video.  If we fail identify the video and request a new random sample.
        try:
            cap = cv.CaptureFromFile(dir)

            length = int(cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_COUNT))
            length = length - 1

            max = length - vid_tensor_container.shape[0] - 1
            start_frame = numpy.random.randint(0,max)

            frame_num = 0
            for i in range(start_frame,start_frame+num_frames,1):
                #print i
                if i%step_size==0 and frame_num < vid_tensor_container.shape[0]:

                    #If we need a random frame then search for one.
                    sample_random=random_frame_sampling
                    while(sample_random):
                        #Try to grab a random frame.
                        try:
                            #Sample a video
                            rand_store, random_vid_name = get_random_frame(full_video_list, total_vid_num)

                            if numpy.any(numpy.isnan(rand_store)):
                                raise Exception()
                            width_offset = (rand_store.shape[1] - width)/2
                            height_offset = (rand_store.shape[2] - height)/2
                            rand_vid_tensor_container[frame_num,  vid_num,:,:,:] = rand_store[:,width_offset:width+width_offset,height_offset:height+height_offset]
                            sample_random = False
                        except:
                            print 'The current video file caused an error'
                            print random_vid_name

                    img = cv.QueryFrame(cap)

                    #If the frame is too small scale it up.
                    if img.width < width or img.height < height:
                        #reshape the image
                        scaling_factor = int(numpy.max(numpy.array([numpy.ceil(height/(img.height*1.0)),numpy.ceil(width/(img.width*1.0))])))
                        # print scaling_factor
                        tmp = cv.CreateImage((img.width*scaling_factor,img.height*scaling_factor),img.depth,color)
                        cv.Resize(img,tmp)
                        img = tmp

                    st = numpy.asarray(cv.GetMat(img))
                    st = numpy.swapaxes(st,0,2)

                    #Check for garbage values in the stored frame.
                    if numpy.any(numpy.isnan(st)):
                        raise Exception()

                    width_offset = (st.shape[1] - width)/2
                    height_offset = (st.shape[2] - height)/2
                    try:
                        vid_tensor_container[frame_num,  vid_num, :, :, :] = st[:,width_offset:width+width_offset,height_offset:height+height_offset]
                    except:
                        print numpy.shape(img)
                        print st.shape
                        print 'failure'
                        raise


                    mask_tensor_container[frame_num, vid_num] = 1.0
                    frame_num += 1
            if get_label:
                labels_tensor_container[vid_num] = label
            vid_num += 1
        except:
            print dir
            print "Unexpected error:", sys.exc_info()[0]
            os.system('rm -f '+ dir)
            if random_vid_sampling==False:
                vid_num += 1


    vid_tensor_container      = pre_process_video_tensor(vid_tensor_container)
    rand_vid_tensor_container = pre_process_video_tensor(rand_vid_tensor_container)

    return vid_tensor_container, rand_vid_tensor_container, mask_tensor_container, labels_tensor_container


"""
Totally ripepd from https://github.com/emansim/unsupervised-videos/blob/master/data_handler.py
"""
class BouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, num_frames=20, batch_size=80, image_size=64, num_digits=2, step_length=.1):
        self.seq_length_ = num_frames
        self.batch_size_ = batch_size
        self.image_size_ = image_size
        self.num_digits_ = num_digits
        self.step_length_ = step_length
        self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        try:
            f = h5py.File('./data/mnist.h5')
        except:
            print 'Please set the correct path to MNIST dataset'
            #sys.exit()
        self.data_ = f['train'].value.reshape(-1, 28, 28)
        self.labels_ = np.array(f['train_labels'])
        self.data_test_ = f['test'].value.reshape(-1, 28, 28)
        self.labels_test_ = np.array(f['test_labels'])
        f.close()
        self.indices_ = np.arange(self.data_.shape[0])
        self.indices_test_ = np.arange(self.data_test_.shape[0])
        self.row_ = [0]
        self.row_test_ = [0]
        np.random.shuffle(self.indices_)
        np.random.shuffle(self.indices_test_)

    def GetBatchSize(self):
        return self.batch_size_

    def GetDims(self):
        return self.frame_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        pass

    def location_class(self, x, y):
        x_ind, y_ind = int(x * 3.), int(y * 3.)
        return y_ind * 3 + x_ind

    def GetRandomTrajectory(self, batch_size, return_location_classes=False):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        location_classes = np.zeros((length, batch_size))
        for i in xrange(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_
            # Bounce off edges.
            for j in xrange(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x
            if return_location_classes:
                location_classes[i] = np.array(
                    [self.location_class(x, y) for y, x in zip(start_y[i], start_x[i])]
                )

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        if return_location_classes:
            return start_y, start_x, location_classes
        else:
            return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)

    """
    Hacked so I can generate single frame batches as well
    """
    def _GetBatch(self, seq_length, verbose=False, train=True, random=False, return_classes=False):
        if return_classes:
            start_y, start_x, location_classes = self.GetRandomTrajectory(
                self.batch_size_ * self.num_digits_,
                return_location_classes=True
            )
        else:
            start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)

        # minibatch data
        data = np.zeros((self.batch_size_, seq_length, self.image_size_, self.image_size_), dtype=np.float32)

        if train:
            DS = self.data_
            LABELS = self.labels_
            inds = self.indices_
            row = self.row_
        else:
            DS = self.data_test_
            LABELS = self.labels_test_
            inds = self.indices_test_
            row = self.row_test_
        digit_classes = []
        if return_classes:
            assert self.num_digits_ == 1, "can only have one digit if classifiying digits"
        for j in xrange(self.batch_size_):
            for n in xrange(self.num_digits_):

                # get random digit from dataset
                ind = inds[row[0]]
                row[0] += 1
                if row[0] == DS.shape[0]:
                    row[0] = 0
                    np.random.shuffle(inds)
                digit_image = DS[ind, :, :]
                if return_classes:
                    digit_class = LABELS[ind]
                    digit_class_t = [digit_class] * self.seq_length_
                    digit_classes.append(digit_class_t)

                #Sample random start points
                init = 0
                final = seq_length
                if random:
                    init = np.random.randint(0, seq_length)
                    final = np.random.randint(init,seq_length)+2
                    final = np.minimum(final,seq_length)

                # generate video
                for i in range(init, final):
                    top    = start_y[i, j * self.num_digits_ + n]
                    left   = start_x[i, j * self.num_digits_ + n]
                    bottom = top  + self.digit_size_
                    right  = left + self.digit_size_
                    data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], digit_image)

        if return_classes:
            return data[:, :, :, :, np.newaxis], location_classes, np.array(digit_classes).T
        else:
            return data[:, :, :, :, np.newaxis], None

    def GetBatch(self, verbose=False, random=False, return_classes=False):
        """
        Generates a video batch
        """
        return self._GetBatch(self.seq_length_, verbose=verbose, random=random, return_classes=return_classes)

    def GetTestBatch(self, verbose=False, random=False, return_classes=False):
        return self._GetBatch(self.seq_length_, verbose=verbose, train=False, random=random, return_classes=return_classes)

    def GetImageBatch(self, verbose=False):
        """
        Generates a batch of images sampled from the video generating
        distribution
        """
        _batch, _ = self._GetBatch(1, verbose=verbose)
        batch = np.squeeze(_batch, axis=1)
        return batch, None

    def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
        output_file1 = None
        output_file2 = None

        if output_file is not None:
            name, ext = os.path.splitext(output_file)
            output_file1 = '%s_original%s' % (name, ext)
            output_file2 = '%s_recon%s' % (name, ext)

        # get data
        data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)
        # get reconstruction and future sequences if exist
        if rec is not None:
            rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
            enc_seq_length = rec.shape[0]
        if fut is not None:
            fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
            if rec is None:
                enc_seq_length = self.seq_length_ - fut.shape[0]
            else:
                assert enc_seq_length == self.seq_length_ - fut.shape[0]

        num_rows = 1
        # create figure for original sequence
        plt.figure(2*fig, figsize=(20, 1))
        plt.clf()
        for i in xrange(self.seq_length_):
            plt.subplot(num_rows, self.seq_length_, i+1)
            plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
            plt.axis('off')
        plt.draw()
        if output_file1 is not None:
            print output_file1
            plt.savefig(output_file1, bbox_inches='tight')

        # create figure for reconstuction and future sequences
        plt.figure(2*fig+1, figsize=(20, 1))
        plt.clf()
        for i in xrange(self.seq_length_):
            if rec is not None and i < enc_seq_length:
                plt.subplot(num_rows, self.seq_length_, i + 1)
                plt.imshow(rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.gray, interpolation="nearest")
            if fut is not None and i >= enc_seq_length:
                plt.subplot(num_rows, self.seq_length_, i + 1)
                plt.imshow(fut[i - enc_seq_length, :, :], cmap=plt.cm.gray, interpolation="nearest")
            plt.axis('off')
        plt.draw()
        if output_file2 is not None:
            print output_file2
            plt.savefig(output_file2, bbox_inches='tight')
        else:
            plt.pause(0.1)


import thread_utils
from threading import Lock
class protected_video_gather:

    def __init__(self, batch_size, full_video_list, full_label_list, vid_shape, mask_shape, mean_image, get_label, resize):


        self.full_video_list = full_video_list
        self.full_label_list = full_label_list
        self.vid_shape = vid_shape
        self.mask_shape = mask_shape
        self.mean_image = mean_image
        self.get_label = get_label
        self.resize = resize
        self.batch_size = batch_size

        self.kf = get_minibatches_idx([self.full_video_list.shape[0]],self.batch_size,shuffle=True)
        self.len = len(self.kf)
        self.l = Lock()

    def get_video(self):
        done = False
        self.l.acquire()
        #print len(self.kf)
        if len(self.kf) > 0:
            _, index = self.kf.pop()
        else:
            self.kf = get_minibatches_idx([self.full_video_list.shape[0]],self.batch_size,shuffle=True)
            _, index = self.kf.pop()
            done = True
        self.l.release()

        return directory_to_numpy(index,
                                  self.full_video_list,
                                  self.full_label_list,
                                  self.vid_shape,
                                  self.mask_shape,
                                  self.get_label,
                                  self.mean_image,
                                  self.resize)
class ImageSequenceLoader:
    def __init__(self, batch_size, folder, image_size, resize=False, reverse=True, greyscale=False):
        self.batch_size = batch_size
        self.folder = folder
        self.image_size = (image_size, image_size)
        self.resize = resize
        self.reverse = reverse
        self.greyscale = greyscale

        self.videos = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))]
        self.videos2images = {}
        self.all_ims = []
        for video in self.videos:
            ims = [os.path.join(video, f) for f in os.listdir(video) if ".png" in f]
            key = lambda f: int(f.split("/")[-1].replace("frame_", "").replace(".png", ""))
            ims.sort(key=key)
            self.videos2images[video] = ims
            self.all_ims.extend(ims)

    def load_seq(self, ims):
        seq = [cv2.imread(f) for f in ims]
        # if we are reversing then do so with prob .5
        if self.reverse and random.random() < .5:
            seq.reverse()
        if self.resize:
            seq_r = [cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA) for img in seq]
            if self.greyscale:
                 seq_r = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[np.newaxis, :, :, np.newaxis] for im in seq_r]
            else:
                seq_r = [s[np.newaxis, :, :, :] for s in seq_r]
        else:
            seq_r = [s[np.newaxis, :, :, :] for s in seq]
        seq_r = np.concatenate(seq_r)
        return seq_r

    def get_batch(self):
        vids = np.random.choice(self.videos, size=self.batch_size, replace=False)
        try:
            batch = np.concatenate([self.load_seq(self.videos2images[v])[np.newaxis, :, :, :, :] for v in vids])
            return batch.astype(np.float) / 255.
        except:
            print("failed to load, retrying...")
            return self.get_batch()

    def get_image_batch(self):
        ims = np.random.choice(self.all_ims, size=self.batch_size, replace=False)
        im_batch = [cv2.imread(f) for f in ims]
        if self.resize:
            im_batch_r = [cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA) for img in im_batch]
            if self.greyscale:
                 im_batch_r = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[np.newaxis, :, :, np.newaxis] for im in im_batch_r]
            else:
                im_batch_r = [s[np.newaxis, :, :, :] for s in im_batch_r]
        else:
            im_batch_r = [s[np.newaxis, :, :, :] for s in im_batch]
        return np.concatenate(im_batch_r).astype(np.float) / 255.


class ImageSequenceDataHandler:
    def __init__(self, batch_size, folder, image_size, resize=False, reverse=True, greyscale=False):
        self.batch_loader = ImageSequenceLoader(batch_size, folder, image_size, resize=resize, reverse=reverse, greyscale=greyscale)

        self.abq = thread_utils.AsynchronousBatchQueue(self.batch_loader.get_batch, 4, 4)

    def get_result(self):

        res = self.abq.get_result()
        return res

class ImageDataHandler:
    def __init__(self, batch_size, folder, image_size, resize=False, greyscale=False):
        self.batch_loader = ImageSequenceLoader(batch_size, folder, image_size, resize=resize, reverse=False, greyscale=greyscale)

        self.abq = thread_utils.AsynchronousBatchQueue(self.batch_loader.get_image_batch, 4, 4)

    def get_result(self):

        res = self.abq.get_result()
        return res

from collections import defaultdict
class ChairTestDataset:
    def __init__(self, batch_size, num_frames, folder, image_size, greyscale=False):
        self.batch_size = batch_size
        self.folder = folder
        self.image_size = (image_size, image_size)
        self.greyscale = greyscale
        self.num_frames = num_frames
        # get videos
        self.videos = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))]

        self.models2images = defaultdict(list)
        self.videos2images = {}
        self.images2models = {}
        self.train_images = []
        self.test_images = []
        for video in self.videos:
            ims = [os.path.join(video, f) for f in os.listdir(video) if ".png" in f]
            key = lambda f: int(f.split("/")[-1].replace("frame_", "").replace(".png", ""))
            ims.sort(key=key)
            self.videos2images[video] = ims

            model = video.split("model_")[1].split("_")[0]
            self.models2images[model].extend(ims)
        # split train / test images for model identification task
        for model in self.models2images.keys():
            ims = [im for im in self.models2images[model]]
            for im in self.models2images[model]:
                self.images2models[im] = model

            random.shuffle(ims)
            l = len(ims)
            self.train_images.extend(ims[:int(l * .9)])
            self.test_images.extend(ims[int(l * .9):])

        self.models2ind = {model: i for i, model in enumerate(self.models2images.keys())}

        # seperate out training and testing videos for rotation prediction
        self.train_videos = [video for video in self.videos]
        random.shuffle(self.train_videos)
        n_train = int(len(self.train_videos) * .9)
        self.train_videos, self.test_videos = self.train_videos[:n_train], self.train_videos[n_train:]

    def load_image(self, imf):
        im = cv2.imread(imf)
        im = cv2.resize(im, self.image_size, interpolation=cv2.INTER_AREA)
        if self.greyscale:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        return im.astype(np.float) / 255.

    def get_id_batch(self, train):
        if train:
            ims = np.random.choice(self.train_images, self.batch_size * self.num_frames, replace=False)
        else:
            ims = np.random.choice(self.test_images, self.batch_size * self.num_frames, replace=False)
        models = [self.images2models[im] for im in ims]
        model_inds = [self.models2ind[model] for model in models]
        images = np.concatenate([self.load_image(im)[np.newaxis, :, :, :] for im in ims])
        images = images.reshape([self.batch_size, self.num_frames, self.image_size[0], self.image_size[1], 1])
        model_inds = np.array(model_inds).reshape([self.batch_size, self.num_frames])
        return images, model_inds










class LSUN_data_handler:
    def __init__(self, db_path, batch_size):
        self.env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
        self.batch_size = batch_size
        self.batch_index = 0
        self.txn = self.env.begin(write=False)
        self.num_ims = self.txn.stat()['entries']
        self.cursor = self.txn.cursor()
        assert self.cursor.next()
        self.num_batches = self.num_ims // batch_size

    def load_batch(self, im_size):
        ims = []

        for i in range(self.batch_size):
            k, v = self.cursor.item()
            assert self.cursor.next(), "screwed up yer inds mang"
            img = cv2.imdecode(np.fromstring(v, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)


            # resize to smallest side is im_size
            h, w, d = img.shape
            r = float(im_size) / min(h, w)
            new_size = min(w, int(r * w)), min(h, int(r * h))
            im_r = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            h, w, d = im_r.shape
            if h <= w:
                buf = (w - im_size) / 2
                im_c = im_r[:, buf:buf+im_size]
            else:
                buf = (h - im_size) / 2
                im_c = im_r[buf:buf+im_size, :]

            im_f = cv2.cvtColor(im_c, cv2.COLOR_BGR2RGB)

            ims.append(im_f[np.newaxis, :, :, :])

        self.batch_index =  (self.batch_index + 1) % self.num_batches
        if self.batch_index == 0:
            print "finished epoch"
            #self.txn = self.env.begin(write=False)
            self.cursor = self.txn.cursor()
            assert self.cursor.next()
        return np.concatenate(ims, axis=0), []


def line2fnameInd(line, folder):
    ls = line.split()
    return os.path.join(folder, ls[0]), int(ls[1])
class imagenet_data_handler:
    def __init__(self, batch_size, data_path="data/imagenet_resized"):
        self.batch_size = batch_size
        self.batch_index = 0
        self.data_path = data_path
        self.train_dir = os.path.join(data_path, "train")
        self.val_dir = os.path.join(data_path, "val")
        self.test_dir = os.path.join(data_path, "test")
        self.label_dir = os.path.join(data_path, "labels")
        self.train_f = os.path.join(self.label_dir, "train.txt")
        self.val_f = os.path.join(self.label_dir, "val.txt")
        self.test_f = os.path.join(self.label_dir, "test.txt")
        with open(self.train_f, 'r') as f:
            self.train = [line2fnameInd(line, self.train_dir) for line in f]
        with open(self.val_f, 'r') as f:
            self.val = [line2fnameInd(line, self.val_dir) for line in f]
        with open(self.test_f, 'r') as f:
            self.test = [line2fnameInd(line, self.test_dir) for line in f]

        np.random.shuffle(self.train)
        np.random.shuffle(self.test)
        np.random.shuffle(self.val)
        self.num_batches = len(self.train) // batch_size
        self.num_val_batches = len(self.val) // batch_size
        self.num_test_batches = len(self.test) // batch_size
        self.num_ims = len(self.train)

    def load_batch(self, im_size, crop_size=None):
        if crop_size is None:
            crop_size = im_size
        ims = []
        labels = []
        start = (im_size / 2) - (crop_size / 2)
        stop = (im_size / 2) + (crop_size / 2)

        batch = self.train[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]
        for (fname, label) in batch:
            img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)


            # resize to smallest side is im_size
            h, w, d = img.shape
            r = float(im_size) / min(h, w)
            new_size = min(w, int(r * w)), min(h, int(r * h))
            im_r = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            h, w, d = im_r.shape
            if h <= w:
                buf = (w - im_size) / 2
                im_c = im_r[:, buf:buf+im_size]
            else:
                buf = (h - im_size) / 2
                im_c = im_r[buf:buf+im_size, :]

            im_f = cv2.cvtColor(im_c, cv2.COLOR_BGR2RGB)

            im_c = im_f[start:stop, start:stop, :]

            ims.append(im_c[np.newaxis, :, :, :])
            labels.append(label)

        self.batch_index =  (self.batch_index + 1) % self.num_batches
        if self.batch_index == 0:
            print "finished epoch"
        return np.concatenate(ims, axis=0), np.array(labels)

class MultiScaleCifar:
    def __init__(self, patch_size, scales):
        """
        produces (patch_size x patch_size) images at scales different scales
        """
        self.patch_size = patch_size
        self.scales = scales
        train_set, valid_set, test_set, self.width, self.height, self.channels, self.num_labels = load_cifar_data()
        self.train_ims, self.train_labels = train_set
        self.valid_ims, self.valid_labels = valid_set
        self.test_ims, self.test_lables = test_set
        self.train_ind = 0
        self.valid_ind = 0
        self.test_ind = 0

    def get_batch(self, ds, scale, size):
        assert scale <= self.scales-1
        if ds == "train":
            if self.train_ind + size > len(self.train_ims):
                self.train_ind = 0
            ims = self.train_ims[self.train_ind:self.train_ind+size, :, :, :]
            self.train_ind = (self.train_ind + size) % len(self.train_ims)
        elif ds == "valid":
            if self.valid_ind + size > len(self.valid_ims):
                self.valid_ind = 0
            ims = self.valid_ims[self.valid_ind:self.valid_ind+size, :, :, :]
            self.valid_ind = (self.valid_ind + size) % len(self.valid_ims)
        elif ds == "test":
            if self.test_ind + size > len(self.test_ims):
                self.test_ind = 0
            ims = self.test_ims[self.test_ind:self.test_ind+size, :, :, :]
            self.test_ind = (self.test_ind + size) % len(self.test_ims)
        else:
            assert False

        patches = []
        for im in ims:
            if self.width == self.patch_size * (2**scale):
                glimpse = im
            else:
                receptive_size = self.patch_size * (2**scale)
                start = self.width - receptive_size
                inds = np.random.randint(0, start, 2)
                glimpse = im[inds[0]:inds[0]+receptive_size, inds[1]:inds[1]+receptive_size]
            patch = cv2.resize(glimpse, (self.patch_size, self.patch_size))
            patches.append(patch)
        return np.array(patches)
    def multiscale_batch(self, ds, batch_size):
        scale_batches = []
        for scale in range(self.scales):
            batch = self.get_batch(ds, scale, batch_size)
            scale_batches.append(batch)
        # shuffle so tensorboard can see all batches
        random.shuffle(scale_batches)
        patch_batch = np.concatenate(scale_batches)
        return patch_batch

