"""
Created on June 4th, 2020
@author: itailang
"""

import tensorflow as tf


class Adversary:
    def __init__(self, adversary_name, batch_size, num_points):
        """Creates adversarial manipulation to a point cloud.
        Arguments:
            adversary_name: A string, name of the adversary
            batch_size: An integer, number of attacked point clouds in a batch.
            num_points: An integer, number of points in each point cloud.
        Inputs:
            input_pc_pl: A `Tensor` of shape (batch_size, num_points, 3), original point cloud.
        Outputs:
            adv_pc: A `Tensor` of shape (batch_size, num_points, 3), adversarial point cloud.
        """

        self._batch_size = batch_size
        self._num_points = num_points
        self.pert = tf.get_variable(name=adversary_name, shape=[self._batch_size, self._num_points, 3],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01), dtype=tf.float32)

    def init_pert(self, sess, stddev=0.0000001, seed=55):
        sess.run(tf.assign(self.pert, tf.truncated_normal([self._batch_size, self._num_points, 3], mean=0, stddev=stddev, seed=seed)))

    def get_pert_value(self, sess):
        return sess.run(self.pert)

    def attack(self, input_pc_pl):
        # adversarial perturbation
        adv_pc = input_pc_pl + self.pert

        return adv_pc

    def get_pert_loss(self, sqrt=True):
        # perturbation l2 constraint
        pert_norm_per_point_square = tf.reduce_sum(tf.square(self.pert), 2)       # sum over point coordinates (x, y, z)
        pert_norm_square = tf.reduce_sum(pert_norm_per_point_square, 1)           # sum over all points in the point cloud

        max_norm_per_point_square = tf.reduce_max(pert_norm_per_point_square, 1)  # max point perturbation

        if sqrt:
            pert_norm = tf.sqrt(pert_norm_square)
            pert_loss = pert_norm

            max_norm = tf.sqrt(max_norm_per_point_square)
            max_loss = max_norm
        else:
            pert_loss = pert_norm_square

            max_loss = max_norm_per_point_square

        return pert_loss, max_loss


if __name__ == '__main__':
    import numpy as np

    batch_size = 2
    num_points = 10
    sqrt = True

    adv = Adversary('adv', batch_size, num_points)

    with tf.Session('') as sess:
        adv.init_pert(sess, stddev=1)
        pert_loss_tf, max_loss_tf = adv.get_pert_loss(sqrt=sqrt)

        pert = adv.get_pert_value(sess)
        pert_loss = pert_loss_tf.eval()
        max_loss = max_loss_tf.eval()

    pert_norm_per_point_square_np = np.sum(np.square(pert), axis=2)
    pert_norm_square_np = np.sum(pert_norm_per_point_square_np, axis=1)

    max_norm_per_point_square_np = np.max(pert_norm_per_point_square_np, axis=1)
    if sqrt:
        pert_norm_np = np.sqrt(pert_norm_square_np)
        pert_loss_np = pert_norm_np

        max_norm_np = np.sqrt(max_norm_per_point_square_np)
        max_loss_np = max_norm_np
    else:
        pert_loss_np = pert_norm_square_np

        max_loss_np = max_norm_per_point_square_np

    diff_pert_loss = pert_loss - pert_loss_np
    diff_max_loss = max_loss - max_loss_np

    print('diff pert loss: ', diff_pert_loss)
    print('diff max loss: ', diff_max_loss)
