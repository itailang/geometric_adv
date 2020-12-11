"""
Created on September 29th, 2020
@author: urikotlicki
"""

# Based on code taken from: https://github.com/charlesq34/pointnet

# import system modules
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import os.path as osp
import sys
import time
import matplotlib.pyplot as plt

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
import classifier.provider as provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log/pointnet', help='Log dir [default: log/pointnet]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=140, help='Epoch to run [default: 140]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--save_model_interval', type=int, default=10, help='Inteval between epochs, used for saving the model [default: 10]')
parser.add_argument('--num_classes', type=int, default=13, help='Number of classes [default: 13]')
parser.add_argument('--train_data', type=str, default='log/autoencoder_victim/eval_train/reconstructions_train_set_13l.npy',
                    help='Path to training data [default: log/autoencoder_victim/eval_train/reconstructions_train_set_13l.npy]')
parser.add_argument('--train_labels', type=str, default='log/autoencoder_victim/eval_train/pc_label_train_set_13l.npy',
                    help='Path to training labels [default: log/autoencoder_victim/eval_train/pc_label_train_set_13l.npy]')
parser.add_argument('--val_data', type=str, default='log/autoencoder_victim/eval_val/reconstructions_val_set_13l.npy',
                    help='Path to validation data [default: log/autoencoder_victim/eval_val/reconstructions_val_set_13l.npy]')
parser.add_argument('--val_labels', type=str, default='log/autoencoder_victim/eval_val/pc_label_val_set_13l.npy',
                    help='Path to validation labels [default: log/autoencoder_victim/eval_val/pc_label_val_set_13l.npy]')
parser.add_argument('--model_path', default=None, help='model checkpoint file path - in case of loading pretrained model to continue training [default: None]')
parser.add_argument('--restore_epoch', type=int, default=0,  help='Restore epoch [default: 0]')
FLAGS = parser.parse_args()

print('Train classifier flags:', FLAGS)


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = osp.join(parent_dir, 'classifier', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not osp.exists(osp.join(parent_dir, LOG_DIR)):
    os.mkdir(osp.join(parent_dir, LOG_DIR))
os.system('cp %s %s' % (osp.join(parent_dir, MODEL_FILE), osp.join(parent_dir,LOG_DIR)))  # bkp of model def
os.system('cp train_classifier.py %s' % osp.join(parent_dir, LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(osp.join(parent_dir, LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = FLAGS.num_classes

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=None)
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(osp.join(parent_dir, LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(osp.join(parent_dir, LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        if FLAGS.model_path is not None:
            saver.restore(sess, osp.join(parent_dir, FLAGS.model_path))
            log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        mean_loss = np.zeros(int(MAX_EPOCH/FLAGS.save_model_interval))
        accuracy = np.zeros(int(MAX_EPOCH/FLAGS.save_model_interval))
        eval_mean_loss = np.zeros(int(MAX_EPOCH/FLAGS.save_model_interval))
        eval_accuracy = np.zeros(int(MAX_EPOCH/FLAGS.save_model_interval))
        eval_avg_class_acc = np.zeros(int(MAX_EPOCH/FLAGS.save_model_interval))

        for epoch in range(FLAGS.restore_epoch, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()
            start_time = time.time()

            [mean_loss_res, accuracy_res] = train_one_epoch(sess, ops, train_writer)
            end_time = time.time()
            print('Epoch training time: %04f minutes' % ((end_time - start_time) / 60.0))

            start_time = time.time()
            [eval_mean_loss_val, eval_accuracy_val, eval_avg_class_acc_val] = eval_one_epoch(sess, ops, test_writer)
            end_time = time.time()
            print('Epoch eval time: %04f minutes' % ((end_time - start_time) / 60.0))
            
            # Save the variables to disk.
            if (epoch+1) % FLAGS.save_model_interval == 0:
                save_path = saver.save(sess, osp.join(parent_dir, LOG_DIR, 'model-%03d.ckpt' % (epoch+1)))
                log_string("Model saved in file: %s" % save_path)
                mean_loss[int(epoch/FLAGS.save_model_interval)] = mean_loss_res
                accuracy[int(epoch/FLAGS.save_model_interval)] = accuracy_res
                eval_mean_loss[int(epoch/FLAGS.save_model_interval)] = eval_mean_loss_val
                eval_accuracy[int(epoch/FLAGS.save_model_interval)] = eval_accuracy_val
                eval_avg_class_acc[int(epoch/FLAGS.save_model_interval)] = eval_avg_class_acc_val

                # Save statistics
                np.save(osp.join(parent_dir, FLAGS.log_dir, 'mean_loss'), mean_loss)
                np.save(osp.join(parent_dir, FLAGS.log_dir, 'accuracy'), accuracy)
                np.save(osp.join(parent_dir, FLAGS.log_dir, 'eval_mean_loss'), eval_mean_loss)
                np.save(osp.join(parent_dir, FLAGS.log_dir, 'eval_accuracy'), eval_accuracy)
                np.save(osp.join(parent_dir, FLAGS.log_dir, 'eval_avg_class_acc'), eval_avg_class_acc)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    current_data = np.load(osp.join(parent_dir, FLAGS.train_data))
    current_label = np.load(osp.join(parent_dir, FLAGS.train_labels))
    current_data = current_data[:, 0:NUM_POINT, :]
    current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        # rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        rotated_data = current_data[start_idx:end_idx, :, :]  # No rotation
        jittered_data = provider.jitter_point_cloud(rotated_data)
        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    mean_loss = loss_sum / float(num_batches)
    accuracy = total_correct / float(total_seen)
    log_string('mean loss: %f' % mean_loss)
    log_string('accuracy: %f' % accuracy)
    return [mean_loss, accuracy]

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    current_data = np.load(osp.join(parent_dir, FLAGS.val_data))
    current_label = np.load(osp.join(parent_dir, FLAGS.val_labels))
    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)

    eval_mean_loss = loss_sum / float(total_seen)
    eval_accuracy = total_correct / float(total_seen)
    eval_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))
    log_string('eval mean loss: %f' % eval_mean_loss)
    log_string('eval accuracy: %f' % eval_accuracy)
    log_string('eval avg class acc: %f' % eval_avg_class_acc)
    return [eval_mean_loss, eval_accuracy, eval_avg_class_acc]


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
