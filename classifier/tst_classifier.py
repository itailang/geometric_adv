"""
Created on September 29th, 2020
@author: urikotlicki
"""

# Based on code taken from: https://github.com/charlesq34/pointnet

# import system modules
import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import os.path as osp
import sys

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import classifier.provider as provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--num_classes', type=int, default=13, help='Number of classes [default: 13]')
parser.add_argument('--model_path', default='log/pointnet/model-150.ckpt', help='model checkpoint file path [default: log/pointnet/model-150.ckpt]')
parser.add_argument('--dump_dir', default='log/pointnet/log_test', help='dump folder path [log/pointnet/test_log]')
parser.add_argument('--test_data', type=str, default='log/autoencoder_victim/eval/point_clouds_test_set_13l.npy',
                    help='Path to test data [default: log/autoencoder_victim/eval/point_clouds_test_set_13l.npy]')
parser.add_argument('--test_labels', type=str, default='log/autoencoder_victim/eval/pc_label_test_set_13l.npy',
                    help='Path to test data [default: log/autoencoder_victim/eval/pc_label_test_set_13l.npy]')
parser.add_argument('--pc_classes', type=str, default='log/autoencoder_victim/eval/pc_classes_13l.npy',
                    help='Path to pc class names [default: log/autoencoder_victim/eval/pc_classes_13l.npy]')
FLAGS = parser.parse_args()

print('Test classifier flags:', FLAGS)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  # import network module
DUMP_DIR = FLAGS.dump_dir
if not osp.exists(osp.join(parent_dir, DUMP_DIR)):
    os.mkdir(osp.join(parent_dir, DUMP_DIR))
LOG_FOUT = open(osp.join(parent_dir, DUMP_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = FLAGS.num_classes
PC_CLASSES = np.load(osp.join(parent_dir, FLAGS.pc_classes))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, osp.join(parent_dir, MODEL_PATH))
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(osp.join(parent_dir, DUMP_DIR, 'pred_label.txt'), 'w')

    current_data = np.load(osp.join(parent_dir, FLAGS.test_data))
    current_label = np.load(osp.join(parent_dir, FLAGS.test_labels))
    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)
    print(current_data.shape)

    file_size = current_data.shape[0]
    assert (file_size % BATCH_SIZE == 0), \
        'The number of examples (%d) should be divided by the batch size (%d) without a remainder' % (file_size, BATCH_SIZE)
    num_batches = file_size // BATCH_SIZE
    if file_size % BATCH_SIZE > 0:
        num_batches = num_batches + 1
        last_batch_partial = True
    else:
        last_batch_partial = False
    print(file_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        # Aggregating BEG
        batch_loss_sum = 0  # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))  # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES))  # 0/1 for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                                                vote_idx / float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        pred_val = np.argmax(batch_pred_sum, 1)
        # Aggregating END

        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum

        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
            fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                
    log_string('test mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('test accuracy: %f' % (total_correct / float(total_seen)))
    log_string('test avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float)

    np.save(osp.join(parent_dir, FLAGS.dump_dir, 'test_accuracy'), (total_correct / float(total_seen)))

    for i, name in enumerate(PC_CLASSES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
