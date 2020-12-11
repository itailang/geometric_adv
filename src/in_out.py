import six
import warnings
import numpy as np
import os
import os.path as osp
import re
from six.moves import cPickle
from multiprocessing import Pool

from external.python_plyfile.plyfile import PlyData

snc_synth_id_to_category = {
    '02691156': 'airplane',     '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',      '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',      '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',       '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',      '02747177': 'trash_bin',  '02942699': 'camera',
    '02954340': 'cap',          '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',        '03207941': 'dishwasher', '03211117': 'display',
    '04379243': 'table',        '04401088': 'telephone',  '02946921': 'can',
    '04460130': 'tower',        '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',     '03325088': 'faucet',     '03337140': 'file_cabinet',
    '03467517': 'guitar',       '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',        '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'loudspeaker',  '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwaves',   '03790512': 'motorbike',  '03797390': 'mug',
    '03928116': 'piano',        '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'flowerpot',    '04004475': 'printer',    '04074963': 'remote',
    '04090263': 'rifle',        '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',         '04330267': 'stove',      '04530566': 'watercraft',
    '04554684': 'washer',       '02858304': 'boat',       '02992529': 'cellphone'
}


def snc_category_to_synth_id():
    d = snc_synth_id_to_category
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    myFile = open(file_name, 'wb')
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()


def unpickle_data(file_name):
    '''Restore data previously saved with pickle_data().
    '''
    inFile = open(file_name, 'rb')
    size = cPickle.load(inFile, fix_imports=True, encoding="latin1")
    for _ in range(size):
        yield cPickle.load(inFile, fix_imports=True, encoding="latin1")
    inFile.close()


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def pc_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standard" folder scheme:
    i.e. /syn_id/model_name.ply
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id


def load_dataset(class_names, set_type, input_dir):
    print('loading shape class %s ' % class_names[0])
    syn_id = snc_category_to_synth_id()[class_names[0]]
    class_dir = osp.join(input_dir, syn_id)
    pc_data_train, pc_data_val, pc_data_test = load_and_split_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

    if set_type == 'train_set':
        pc_data = pc_data_train
    elif set_type == 'val_set':
        pc_data = pc_data_val
    else:
        pc_data = pc_data_test

    slice_idx = [0, len(pc_data.point_clouds)]
    pc_label = [0] * len(pc_data.point_clouds)

    for i in range(1, len(class_names)):
        print('loading shape class %s ' % class_names[i])
        syn_id = snc_category_to_synth_id()[class_names[i]]
        class_dir = osp.join(input_dir, syn_id)
        pc_data_train_curr, pc_data_val_curr, pc_data_test_curr = load_and_split_all_point_clouds_under_folder(
            class_dir, n_threads=8, file_ending='.ply', verbose=True)

        if set_type == 'train_set':
            pc_data_curr = pc_data_train_curr
        elif set_type == 'val_set':
            pc_data_curr = pc_data_val_curr
        else:
            pc_data_curr = pc_data_test_curr

        pc_data.merge(pc_data_curr)

        slice_idx.append(slice_idx[-1] + len(pc_data_curr.point_clouds))
        pc_label += [i] * len(pc_data_curr.point_clouds)

    return pc_data, slice_idx, pc_label


def load_all_point_clouds_under_folder(top_dir, n_threads=20, file_ending='.ply', verbose=False):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader, verbose=verbose)
    return PointCloudDataSet(pclouds, labels=syn_ids + '_' + model_ids, init_shuffle=False)


def load_and_split_all_point_clouds_under_folder(top_dir, n_threads=20, file_ending='.ply',
                                                 split=(.85, .05, .10), seed=42, verbose=False):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader, verbose=verbose)

    pclouds_train, pclouds_val, pclouds_test, perm = split_data(pclouds, split, seed)
    model_ids_train, model_ids_val, model_ids_test, _ = split_data(model_ids, split, seed, perm)
    syn_ids_train, syn_ids_val, syn_ids_test, _ = split_data(syn_ids, split, seed, perm)

    ds_train = PointCloudDataSet(pclouds_train, labels=syn_ids_train + '_' + model_ids_train, init_shuffle=False)
    ds_val = PointCloudDataSet(pclouds_val, labels=syn_ids_val + '_' + model_ids_val, init_shuffle=False)
    ds_test = PointCloudDataSet(pclouds_test, labels=syn_ids_test + '_' + model_ids_test, init_shuffle=False)

    return ds_train, ds_val, ds_test


def load_point_clouds_from_filenames(file_names, n_threads, loader, verbose=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids


def split_data(data, split, seed, perm=None):
    assert sum(split) == 1., ('data split does not sum to 1: %.2f' % sum(split))

    num_examples = data.shape[0]
    if perm is not None:
        assert perm.shape[0] == data.shape[0], ('perm.shape: %s data.shape: %s' % (perm.shape, data.shape))
    else:
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(num_examples)
        np.random.shuffle(perm)

    data = data[perm]
    train_end = int(split[0] * num_examples + 0.5)             # add 0.5 and use int() to round a remainder of 0.5 upward instead of downward
    val_end = int((split[0] + split[1]) * num_examples + 0.5)  # add 0.5 and use int() to round a remainder of 0.5 upward instead of downward

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    counts = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
    assert sum(counts) == num_examples,\
        ('data split (%d, %d, %d) does not sum to num_examples (%d)' % (counts[0], counts[1], counts[2], num_examples))

    return train_data, val_data, test_data, perm


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None, copy=True, init_shuffle=True):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''

        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels

        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if noise is not None:
            assert (type(noise) is np.ndarray)
            if copy:
                self.noisy_point_clouds = noise.copy()
            else:
                self.noisy_point_clouds = noise
        else:
            self.noisy_point_clouds = None

        if copy:
            self.point_clouds = point_clouds.copy()
        else:
            self.point_clouds = point_clouds

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]
        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = self.noisy_point_clouds[perm]
        return self

    def shuffle_points(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.n_points)
        for i in range(self.num_examples):
            #print('Shuffle point cloud %d/%d' % (i+1, self.num_examples))
            np.random.shuffle(perm)
            self.point_clouds[i, :, :] = self.point_clouds[i, perm, :]
            if self.noisy_point_clouds is not None:
                self.noisy_point_clouds[i, :, :] = self.noisy_point_clouds[i, perm, :]
        return self

    def next_batch(self, batch_size, shuffle=True, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            if shuffle:
                self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.noisy_point_clouds is None:
            return self.point_clouds[start:end], self.labels[start:end], None
        else:
            return self.point_clouds[start:end], self.labels[start:end], self.noisy_point_clouds[start:end]

    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = self.point_clouds[perm]
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = self.noisy_point_clouds[perm]
        return pc, lb, ns

    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.point_clouds = np.vstack((self.point_clouds, other_data_set.point_clouds))

        labels_1 = self.labels.reshape([self.num_examples, 1])  # TODO = move to init.
        labels_2 = other_data_set.labels.reshape([other_data_set.num_examples, 1])
        self.labels = np.vstack((labels_1, labels_2))
        self.labels = np.squeeze(self.labels)

        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = np.vstack((self.noisy_point_clouds, other_data_set.noisy_point_clouds))

        self.num_examples = self.point_clouds.shape[0]

        return self
