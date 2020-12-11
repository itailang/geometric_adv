"""
Created on June 4th, 2020
@author: itailang
"""

import time
import tensorflow as tf
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from tflearn import is_training

from src.adversary_autoencoder import AdversaryAutoEncoder
from src.adversary import Adversary
from src.general_utils import plot_3d_point_cloud

try:
    from external.structural_losses.tf_nndistance import nn_distance
    from external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
    

class AdvAE(AdversaryAutoEncoder):
    """
    Adversary for auto-encoder of point-clouds.
    """

    def __init__(self, adversary_name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AdversaryAutoEncoder.__init__(self, adversary_name, graph, configuration)

        with tf.variable_scope(adversary_name):
            self.adversary = Adversary('pert', c.batch_size, c.n_input[0])
            self.adv = self.adversary.attack(self.x)

        with tf.variable_scope(c.ae_name):
            self.z = c.encoder(self.adv, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])

        with tf.variable_scope(adversary_name):
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

            self.restore_ae_model(c.ae_dir, c.ae_name, c.ae_restore_epoch, verbose=True)

    def _create_loss(self):
        c = self.configuration

        self.loss_ae, self.loss_ae_per_pc = self._create_ae_loss(c.loss)  # for attack success metric and/or loss
        self.input_dist, self.input_dist_per_pc, self.max_dist, self.max_dist_per_pc = self._create_input_dist(c.loss)  # for attack success metric and/or loss

        # target adversarial loss
        if c.loss_adv_type == 'latent':
            self.loss_adv = self._create_latent_loss(self.z, self.target_z)
        else:
            self.loss_adv = self.loss_ae_per_pc

        # source distance loss
        self.loss_pert, self.loss_max = self.adversary.get_pert_loss()

        if c.loss_dist_type == 'pert':
            if c.max_point_pert_weight > 0.0:
                self.loss_dist = self.loss_pert + c.max_point_pert_weight * self.loss_max
            else:
                self.loss_dist = self.loss_pert
        else:
            if c.max_point_dist_weight > 0.0:
                self.loss_dist = self.input_dist_per_pc + c.max_point_dist_weight * self.max_dist_per_pc
            else:
                self.loss_dist = self.input_dist_per_pc

        # total loss
        self.loss = tf.reduce_sum(self.loss_adv + tf.multiply(self.dist_weight, self.loss_dist))

    def _create_latent_loss(self, latent_source, latent_target, sqrt=True):
        diff = latent_source - latent_target
        diff_norm_square = tf.reduce_sum(tf.square(diff), axis=1)
        if sqrt:
            diff_norm = tf.sqrt(diff_norm_square)
            loss_latent = diff_norm
        else:
            loss_latent = diff_norm_square

        return loss_latent

    def _create_ae_loss(self, loss_type):
        if loss_type == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            loss_ae_per_pc = tf.reduce_mean(cost_p1_p2, axis=1) + tf.reduce_mean(cost_p2_p1, axis=1)
        elif loss_type == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            loss_ae_per_pc = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match), axis=1)

        loss_ae = tf.reduce_mean(loss_ae_per_pc)
        return loss_ae, loss_ae_per_pc

    def _create_input_dist(self, loss_type):
        if loss_type == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.adv, self.x)
            input_dist_per_pc = tf.reduce_mean(cost_p1_p2, axis=1) + tf.reduce_mean(cost_p2_p1, axis=1)
            max_dist_per_pc = tf.reduce_max(cost_p1_p2, axis=1)
        elif loss_type == 'emd':
            match = approx_match(self.adv, self.x)
            m_cost = match_cost(self.adv, self.x, match)
            input_dist_per_pc = tf.reduce_mean(m_cost, axis=1)
            max_dist_per_pc = tf.reduce_max(m_cost, axis=1)

        input_dist = tf.reduce_mean(input_dist_per_pc)
        max_dist = tf.reduce_mean(max_dist_per_pc)
        return input_dist, input_dist_per_pc, max_dist, max_dist_per_pc

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.attack_op = self.optimizer.minimize(self.loss, var_list=self.adversary.pert)

    def attack(self, source_pc, target_latent, target_pc, target_ae_loss_ref, configuration, log_file=None):
        n_examples = len(source_pc)
        adversarial_metrics = []
        adversarial_pc_input = []
        adversarial_pc_recon = []
        batch_size = configuration.batch_size

        assert n_examples % batch_size == 0, 'The number of examples (%d) should be divided by the batch size (%d)' % (n_examples, batch_size)
        n_batches = n_examples // batch_size

        # Loop over all batches
        for i in range(n_batches):
            start_time = time.time()

            s_idx = i*batch_size
            e_idx = (i+1)*batch_size
            adversarial_metrics_batch, adversarial_pc_input_batch, adversarial_pc_recon_batch =\
                self._attack_one_batch(source_pc[s_idx:e_idx], target_latent[s_idx:e_idx], target_pc[s_idx:e_idx], target_ae_loss_ref[s_idx:e_idx], log_file)

            # aggregate results
            adversarial_metrics.append(adversarial_metrics_batch)
            adversarial_pc_input.append(adversarial_pc_input_batch)
            adversarial_pc_recon.append(adversarial_pc_recon_batch)

            duration = time.time() - start_time

            print("Batch: %04d out of %04d, attack time (minutes): %.4f" % (i+1, n_batches, duration / 60.0))
            if log_file is not None:
                log_file.write('Batch %04d\tDuration %.4f\n' % (i+1, duration / 60.0))

        adversarial_metrics_epoch = np.concatenate(adversarial_metrics, axis=1)
        adversarial_pc_input_epoch = np.concatenate(adversarial_pc_input, axis=1)
        adversarial_pc_recon_epoch = np.concatenate(adversarial_pc_recon, axis=1)

        return adversarial_metrics_epoch, adversarial_pc_input_epoch, adversarial_pc_recon_epoch

    def _attack_one_batch(self, source_pc, target_latent, target_pc, target_ae_loss_ref, log_file=None):
        c = self.configuration

        dist_weight_list = c.dist_weight_list
        num_dist_weight = len(dist_weight_list)

        target_recon_error_agg = 1e10 * np.ones(shape=(num_dist_weight, c.batch_size)).astype(np.float32)
        adversarial_metrics_agg = np.zeros(shape=(num_dist_weight, c.batch_size, 4)).astype(np.float32)  # loss_adv, loss_dist, source_chamfer_dist, target_nre
        adversarial_pc_input_agg = np.zeros(shape=(num_dist_weight, c.batch_size, c.n_input[0], 3)).astype(np.float32)
        adversarial_pc_recon_agg = np.zeros(shape=(num_dist_weight, c.batch_size, c.n_output[0], 3)).astype(np.float32)

        feed_dict = {self.x: source_pc, self.target_z: target_latent, self.gt: target_pc}

        if c.loss_dist_type == 'pert':
            loss_max_tf = self.loss_max
        else:
            loss_max_tf = self.max_dist_per_pc

        for i, dist_weight in enumerate(dist_weight_list):
            is_training(False, session=self.sess)

            # start attack (for a given dist_weight)
            feed_dict[self.dist_weight] = np.ones(c.batch_size) * dist_weight
            self.adversary.init_pert(self.sess)

            for iteration in range(c.num_iterations):
                _ = self.sess.run([self.attack_op], feed_dict=feed_dict)

                loss_adv, loss_dist, loss_pert, loss_max, source_chamfer_dist, target_recon_error =\
                    self.sess.run([self.loss_adv, self.loss_dist, self.loss_pert, loss_max_tf,
                                   self.input_dist_per_pc, self.loss_ae_per_pc], feed_dict=feed_dict)

                loss = loss_adv + dist_weight * loss_dist
                if (iteration+1) % ((c.num_iterations // 10) or 1) == 0:
                    print("Weight {} of {}, Iteration {} of {}, loss={} loss_adv={} loss_dist={} loss_pert={} loss_max={}".format(
                        i+1, num_dist_weight, iteration+1, c.num_iterations, np.mean(loss), np.mean(loss_adv),
                        np.mean(loss_dist), np.mean(loss_pert), np.mean(loss_max)))
                    if log_file is not None:
                        log_file.write(
                            'Dist weight %.4f\tIteration %.04d\tloss: %.4f\tloss_adv: %.4f\tloss_dist: %.4f\tloss_pert: %.4f\tloss_max: %.4f\n' %
                            (dist_weight, iteration+1, np.mean(loss), np.mean(loss_adv), np.mean(loss_dist), np.mean(loss_pert), np.mean(loss_max)))

                # update output data in case of lower target reconstruction error (for each instance)
                if (iteration+1) >= c.num_iterations_thresh:
                    adv_pc_input = self.sess.run(self.adv, feed_dict=feed_dict)
                    adv_pc_recon, _ = self.reconstruct(source_pc, compute_loss=False)

                    for j in range(c.batch_size):
                        if target_recon_error[j] < target_recon_error_agg[i, j]:
                            target_recon_error_agg[i, j] = target_recon_error[j]
                            target_nre = target_recon_error[j]/target_ae_loss_ref[j]  # normalized reconstruction error

                            adversarial_metrics_agg[i, j] = [loss_adv[j], loss_dist[j], source_chamfer_dist[j], target_nre]

                            adversarial_pc_input_agg[i, j] = adv_pc_input[j]
                            adversarial_pc_recon_agg[i, j] = adv_pc_recon[j]
            # end attack

        adversarial_metrics_agg_all = np.concatenate([adversarial_metrics_agg, np.expand_dims(target_recon_error_agg, axis=-1)], axis=-1)

        return adversarial_metrics_agg_all, adversarial_pc_input_agg, adversarial_pc_recon_agg

    def _attack_one_batch_binary_step(self, attacked_data):
        c = self.configuration
        is_training(True, session=self.sess)

        # the bound for the binary search
        lower_bound = np.zeros(c.batch_size)
        dist_weight = np.ones(c.batch_size) * c.init_dist_weight
        upper_bound = np.ones(c.batch_size) * c.upper_bound_dist_weight

        out_best_adv = [1e10] * c.batch_size
        out_best_dist = [1e10] * c.batch_size
        out_best_attack = np.ones(shape=(c.batch_size, c.n_input[0], 3))

        feed_dict = {self.x: attacked_data, self.dist_weight: dist_weight}

        for out_step in range(c.binary_search_step):
            feed_dict[self.dist_weight] = dist_weight
            self.adversary.init_pert(self.sess)
            best_adv = [1e10] * c.batch_size
            best_dist = [1e10] * c.batch_size

            for iteration in range(c.num_iterations):
                _ = self.sess.run([self.attack_op], feed_dict=feed_dict)

                loss_adv, loss_dist, adv_pc = self.sess.run([self.loss_adv, self.loss_dist, self.adv], feed_dict=feed_dict)
                loss = np.mean(loss_adv + dist_weight * loss_dist)
                if iteration % ((c.num_iterations // 10) or 1) == 0:
                    print("Iteration {} of {}: loss={} loss_adv={} loss_dist={}".format(
                        iteration, c.num_iterations, loss, np.mean(loss_adv), np.mean(loss_dist)))

                for e, (adv, dist, pc) in enumerate(zip(loss_adv, loss_dist, adv_pc)):
                    if dist < best_dist[e]:
                        best_dist[e] = dist
                        best_adv[e] = adv
                    if dist < out_best_dist[e]:
                        out_best_dist[e] = dist
                        out_best_adv[e] = adv
                        out_best_attack[e] = pc

            # adjust the constant as needed
            for e in range(c.batch_size):
                if best_dist[e] <= out_best_dist[e]:
                    # success
                    lower_bound[e] = max(lower_bound[e], dist_weight[e])
                    dist_weight[e] = (lower_bound[e] + upper_bound[e]) / 2
                    # print('new result found!')
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], dist_weight[e])
                    dist_weight[e] = (lower_bound[e] + upper_bound[e]) / 2

        return out_best_adv, out_best_dist, out_best_attack, dist_weight
