"""
Created on June 4th, 2020
@author: itailang
"""

import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from tflearn import is_training

from src.general_utils import apply_augmentations, iterate_in_chunks
from src.neural_net import NeuralNet, MODEL_SAVER_ID


class AdversaryAutoEncoder(NeuralNet):
    '''Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.'''

    def __init__(self, name, graph, configuration):
        NeuralNet.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        batch_size = configuration.batch_size  # instead of None

        in_shape = [batch_size] + self.n_input
        out_shape = [batch_size] + self.n_output

        target_z_shape = [batch_size] + [configuration.encoder_args['n_filters'][-1]]
        dist_weight_shape = [batch_size]

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            self.gt = tf.placeholder(tf.float32, out_shape)

            # for adversarial attack
            self.target_z = tf.placeholder(tf.float32, target_z_shape)
            self.dist_weight = tf.placeholder(tf.float32, dist_weight_shape)

    def restore_ae_model(self, ae_model_path, ae_name, epoch, verbose=False):
        '''Restore all the variables of a saved ae model.'''
        global_vars = tf.global_variables()
        ae_params = [v for v in global_vars if v.name.startswith(ae_name)]

        saver_ae = tf.train.Saver(var_list=ae_params)
        saver_ae.restore(self.sess, osp.join(ae_model_path, MODEL_SAVER_ID + '-' + str(int(epoch))))

        if verbose:
            print('AE Model restored from %s, in epoch %d' % (ae_model_path, epoch))

    def partial_fit(self, X, GT=None):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        is_training(True, session=self.sess)
        try:
            if GT is not None:
                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT})
            else:
                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X})

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss

    def reconstruct(self, X, GT=None, compute_loss=True, loss_per_pc=False):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if compute_loss:
            if loss_per_pc:
                loss = self.loss_ae_per_pc
            else:
                loss = self.loss_ae
        else:
            loss = self.no_op

        x_reconstr = self.x_reconstr

        if GT is None:
            return self.sess.run((x_reconstr, loss), feed_dict={self.x: X})
        else:
            return self.sess.run((x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

    def get_ae_loss(self, X, GT=None):
        if GT is None:
            feed_dict = {self.x: X}
        else:
            feed_dict = {self.x: X, self.gt: GT}

        return self.sess.run(self.loss_ae, feed_dict=feed_dict)

    def get_ae_loss_per_pc(self, feed_data, orig_data=None):
        feed_data_shape = feed_data.shape
        assert len(feed_data_shape) == 3, 'The feed data should have 3 dimensions'

        if orig_data is not None:
            assert feed_data_shape == orig_data.shape, 'The feed data and original data should have the same size'
        else:
            orig_data = feed_data

        n_examples = feed_data_shape[0]
        ae_loss = np.zeros(n_examples)
        for i in range(0, n_examples, 1):
            ae_loss[i] = self.get_ae_loss(feed_data[i:i+1], orig_data[i:i+1])

        return ae_loss

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def interpolate(self, x, y, steps):
        ''' Interpolate between and x and y input vectors in latent space.
        x, y np.arrays of size (n_points, dim_embedding).
        '''
        in_feed = np.vstack((x, y))
        z1, z2 = self.transform(in_feed.reshape([2] + self.n_input))
        all_z = np.zeros((steps + 2, len(z1)))

        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_z[i, :] = (alpha * z2) + ((1.0 - alpha) * z1)

        return self.sess.run((self.x_reconstr), {self.z: all_z})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def evaluate(self, in_data, configuration, ret_pre_augmentation=False):
        n_examples = in_data.num_examples
        data_loss = 0.
        pre_aug = None
        if self.is_denoising:
            original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
            if ret_pre_augmentation:
                pre_aug = feed_data.copy()
            if feed_data is None:
                feed_data = original_data
            feed_data = apply_augmentations(feed_data, configuration)  # This is a new copy of the batch.
        else:
            original_data, ids, _ = in_data.full_epoch_data(shuffle=False)
            feed_data = apply_augmentations(original_data, configuration)

        b = configuration.batch_size
        reconstructions = np.zeros([n_examples] + self.n_output)
        for i in range(0, n_examples, b):
            if self.is_denoising:
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b], original_data[i:i + b], sort_reconstr=True)
            else:
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b], sort_reconstr=True)

            # Compute average loss
            data_loss += (loss * len(reconstructions[i:i + b]))
        data_loss /= float(n_examples)

        print("evaluation loss=", "{:.9f}".format(data_loss))

        if pre_aug is not None:
            return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data), pre_aug
        else:
            return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data)
        
    def embedding_at_tensor(self, dataset, conf, feed_original=True, apply_augmentation=False, tensor_name='bottleneck'):
        '''
        Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
        Observation: the next layer after latent (z) might be something interesting.
        tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
        '''
        batch_size = conf.batch_size
        original, ids, noise = dataset.full_epoch_data(shuffle=False)

        if feed_original:
            feed = original
        else:
            feed = noise
            if feed is None:
                feed = original

        feed_data = feed
        if apply_augmentation:
            feed_data = apply_augmentations(feed, conf)

        embedding = []
        if tensor_name == 'bottleneck':
            for b in iterate_in_chunks(feed_data, batch_size):
                embedding.append(self.transform(b.reshape([len(b)] + conf.n_input)))
        else:
            embedding_tensor = self.graph.get_tensor_by_name(tensor_name)
            for b in iterate_in_chunks(feed_data, batch_size):
                codes = self.sess.run(embedding_tensor, feed_dict={self.x: b.reshape([len(b)] + conf.n_input)})
                embedding.append(codes)

        embedding = np.vstack(embedding)
        return feed, embedding, ids
        
    def get_reconstructions(self, pclouds, batch_size=50):
        ''' Convenience wrapper of self.reconstruct to get reconstructions of input point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        '''

        reconstructions = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            rcon, _ = self.reconstruct(pclouds[b], compute_loss=False)
            reconstructions.append(rcon)
        return np.vstack(reconstructions)

    def get_latent_vectors(self, pclouds, batch_size=50):
        ''' Convenience wrapper of self.transform to get the latent (bottle-neck) codes for a set of input point 
        clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        '''
        latent_codes = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            latent_codes.append(self.transform(pclouds[b]))
        return np.vstack(latent_codes)
