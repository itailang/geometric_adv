"""
Created on September 21st, 2020
@author: urikotlicki
"""

# Based on code taken from: https://github.com/charlesq34/pointnet

# import system modules
import tensorflow as tf
import numpy as np
import importlib
import os.path as osp


class PointNetClassifier:
    def __init__(self, classifier_path, restore_epoch, num_points, batch_size, num_classes):
        self.BATCH_SIZE = batch_size
        if restore_epoch < 10:
            self.MODEL_PATH = osp.join(classifier_path, 'model-00%d.ckpt' % restore_epoch)
        elif restore_epoch < 100:
            self.MODEL_PATH = osp.join(classifier_path, 'model-0%d.ckpt' % restore_epoch)
        else:
            self.MODEL_PATH = osp.join(classifier_path, 'model-%d.ckpt' % restore_epoch)
        self.GPU_INDEX = 0
        self.MODEL = importlib.import_module('pointnet_cls')  # import network module
        self.NUM_CLASSES = num_classes
        self.NUM_POINTS = num_points
        with tf.device('/gpu:' + str(self.GPU_INDEX)):
            pointclouds_pl, labels_pl = self.MODEL.placeholder_inputs(self.BATCH_SIZE, self.NUM_POINTS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # simple model
            self.pred, self.end_points = self.MODEL.get_model(pointclouds_pl, is_training_pl, self.NUM_CLASSES)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        self.sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(self.sess, self.MODEL_PATH)

        self.ops = {'pointclouds_pl': pointclouds_pl, 'labels_pl': labels_pl, 'is_training_pl': is_training_pl, 'pred': self.pred}

    @staticmethod
    def log_string(out_str):
        print(out_str)

    def classify(self, current_data):
        is_training = False

        num_examples = current_data.shape[0]
        assert num_examples % self.BATCH_SIZE == 0, 'The number of examples (%d) should be divided by the batch size (%d)' % (num_examples, self.BATCH_SIZE)
        num_batches = num_examples // self.BATCH_SIZE

        pred_label = np.zeros(current_data.shape[0], dtype=np.int8)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.BATCH_SIZE
            end_idx = (batch_idx+1) * self.BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            pred_label_batch = np.zeros(cur_batch_size, dtype=np.int8)  # class labels for batch
            feed_dict = {self.ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :], self.ops['is_training_pl']: is_training}
            pred_val_batch = self.sess.run(self.ops['pred'], feed_dict=feed_dict)  # score for classes
            pred_label_batch = np.argmax(pred_val_batch, axis=1).astype(np.int8)
            pred_label[start_idx:end_idx] = pred_label_batch

        return pred_label
