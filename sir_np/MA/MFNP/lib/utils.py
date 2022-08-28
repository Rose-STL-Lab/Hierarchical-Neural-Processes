import logging
import numpy as np
import os
import scipy.sparse as sp
import sys
import tensorflow as tf
import random
from scipy.sparse import linalg

class DataLoader(object):
    def __init__(self, xs, ys, adj_mxs, batch_size, pad_with_last_sample=True, shuffle=False, p_larger_size=None):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        current_size = len(xs)
        if p_larger_size is not None:
            if current_size < p_larger_size:
                xs = np.pad(xs, ((0,p_larger_size-current_size),(0, 0),(0, 0)), 'wrap')
                ys = np.pad(ys, ((0,p_larger_size-current_size),(0, 0),(0, 0)), 'wrap')
                adj_mxs = np.pad(adj_mxs, ((0,p_larger_size-current_size),(0, 0),(0, 0)), 'wrap')

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            xs = np.pad(xs, ((0,num_padding),(0, 0),(0, 0)), 'wrap')
            ys = np.pad(ys, ((0,num_padding),(0, 0),(0, 0)), 'wrap')
            adj_mxs = np.pad(adj_mxs, ((0,num_padding),(0, 0),(0, 0)), 'wrap')
            # x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            # y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            # adj_mxs_padding = np.repeat(adj_mxs[-1:], num_padding, axis=0)
            # xs = np.concatenate([xs, x_padding], axis=0)
            # ys = np.concatenate([ys, y_padding], axis=0)
            # adj_mxs = np.concatenate([adj_mxs, adj_mxs_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys, adj_mxs = xs[permutation], ys[permutation], adj_mxs[permutation]
        self.xs = xs
        self.ys = ys
        self.adj_mxs = adj_mxs

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                adj_i = self.adj_mxs[start_ind: end_ind, ...]
                yield (x_i, y_i, adj_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


# def get_total_trainable_parameter_size():
#     """
#     Calculates the total number of trainable parameters in the current graph.
#     :return:
#     """
#     total_parameters = 0
#     for variable in tf.trainable_variables():
#         # shape is an array of tf.Dimension
#         total_parameters += np.product([x.value for x in variable.get_shape()])
#     return total_parameters

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    #previous
    # x_train_l1: (109* 30, 50, 18, 3), x_train_l2: (80* 30, 50, 85, 3), 
    # x_val: (14* 30, 50, 85, 3), x_test: (15* 30, 50, 85, 3)
    # graph_train_l1 output: (109 * 30, 50, 18, 18)

    # updated
    # x_train_l1: (109* 30, 18, 3), x_train_l2: (80* 30, 85, 3), 
    # x_val: (15* 30, 85, 3), x_test: (15* 30, 85, 3)
    # y_val: (15* 30, 85, 50)
    # graph_train_l1 output: (109 * 30, 18, 18)
    for category in ['train_l1', 'train_l2', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = np.log(cat_data['y']+10.)
        data['graph_' + category] = cat_data['graph']
    l1_x_scaler = StandardScaler(mean=np.mean(data['x_train_l1'],(0,1)), std= np.std(data['x_train_l1'],(0,1)))
    l2_x_scaler = StandardScaler(mean=np.mean(data['x_train_l2'],(0,1)), std= np.std(data['x_train_l2'],(0,1)))
    # l1_y_scaler = StandardScaler(mean=np.mean(data['y_train_l1'],(0,1)), std= np.std(data['y_train_l1'],(0,1)))
    # l2_y_scaler = StandardScaler(mean=np.mean(data['y_train_l2'],(0,1)), std= np.std(data['y_train_l2'],(0,1)))

    data['x_train_l1'] = l1_x_scaler.transform(data['x_train_l1'])
    # data['y_train_l1'] = l1_y_scaler.transform(data['y_train_l1'])
    for category in ['train_l2', 'val', 'test']:
        data['x_' + category] = l2_x_scaler.transform(data['x_' + category])
        # data['y_' + category] = l2_y_scaler.transform(data['y_' + category])

    data['l1_train_loader'] = DataLoader(data['x_train_l1'], data['y_train_l1'], data['graph_train_l1'], batch_size, shuffle=True, p_larger_size=len(data['x_train_l2']))
    data['l2_train_loader'] = DataLoader(data['x_train_l2'], data['y_train_l2'], data['graph_train_l2'], batch_size, shuffle=True, p_larger_size=len(data['x_train_l1']))
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], data['graph_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], data['graph_test'], test_batch_size, shuffle=False)
    data['l1_x_scaler'] = l1_x_scaler
    data['l2_x_scaler'] = l2_x_scaler
    # data['l1_y_scaler'] = l1_y_scaler
    # data['l2_y_scaler'] = l2_y_scaler


    return data

