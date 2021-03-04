"""
@author Nicolas Tapia
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import numpy as np
import tensorflow as tf  # TF 1.x


class MatrixEstimator(object):
    """ TensorFlow 1.x implementation of estimator proposed in:

    Giraldo, L.G.S., Rao, M., & Principe, J.C. (2015). Measures of entropy
    from data using infinitely divisible kernels. IEEE Transactions on
    Information Theory, 61(1), 535-548.

    with modifications made by the authors to attenuate scale and dimension
    related artifacts in the gaussian kernel.

    The originally proposed kernel width formula is:
    sigma = gamma * n ^ (-1 / (4+d))
    with gamma being some empirical constant.

    If normalize_scale is set to True, then the variable is first normalized
    to zero mean and unit variance, as:

    x -> (x - mean) / sqrt(var + epsilon)

    which is equivalent to add the standard deviation as a multiplicative
    dependence in sigma, as done in the classical Silverman rule. This is done
    to achieve invariance to changes of scale during the mutual information
    estimation process. Epsilon is a small number to avoid division by zero.

    If normalize_dimension is set to True, then sigma is computed as:

    sigma = gamma * sqrt(d) * n ^ (-1 / (4+d))

    This is done to center the distribution of pair-wise distances to the same
    mean across variables with different dimensions, and as a consequence
    to attenuate dimension related artifacts.

    Note that normalize_scale=False and normalize_dimension=False will give you
    the original version of the estimator.

    The estimator with those modifications was used in:

    Tapia, N. I. & Estevez, P.A., "On the Information Plane of Autoencoders,"
    in 2020 International Joint Conference on Neural Networks (IJCNN).
    Full text available at: https://arxiv.org/abs/2005.07783

    If you find this software useful, please consider citing our work.
    """

    def __init__(
            self,
            gamma=1.0,
            alpha=1.01,
            epsilon=1e-8,
            normalize_scale=True,
            normalize_dimension=True,
            log_base=2,
            use_memory_efficient_gram=False,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.normalize_scale = normalize_scale
        self.normalize_dimension = normalize_dimension
        self.log_base = log_base
        self.use_memory_efficient_gram = use_memory_efficient_gram

    def _compute_sigma(self, x):
        x_dims = tf.shape(x)
        n = tf.cast(x_dims[0], tf.float32)
        d = tf.cast(x_dims[1], tf.float32)
        sigma = self.gamma * n ** (-1 / (4 + d))
        if self.normalize_dimension:
            sigma = sigma * tf.sqrt(d)
        return sigma

    def _normalize_variable(self, x, x_is_image):
        if x_is_image:
            mean_x = tf.reduce_mean(x)
            var_x = tf.reduce_mean(tf.square(x - mean_x))
        else:
            mean_x, var_x = tf.nn.moments(x, [0])
        std_x = tf.sqrt(var_x + self.epsilon)
        x = (x - mean_x) / std_x
        return x

    def normalized_gram(self, x, sigma_x=None, x_is_image=False):
        """If sigma_x is provided, then that value will be used. Otherwise,
        it will be automatically computed using the formula.
        If x_is_image is True, then the normalization of scale (if applicable)
        is done aggregating all dimensions. If false, each dimension is
        normalized independently.
        """
        if sigma_x is None:
            sigma_x = self._compute_sigma(x)
        if self.normalize_scale:
            x = self._normalize_variable(x, x_is_image)

        # Compute pairwise distances (distance matrix)
        if self.use_memory_efficient_gram:
            # This option stores a smaller tensor in memory, which might be more convenient for you
            # when the dimensionality of the variable is too large, at the cost of introducing some
            # rounding errors due to the intermediate steps
            # (although I expect them to be insignificant in most cases),
            # because it performs
            # (N, Dim) matmul (Dim, N) = (N, N)
            # thanks to an equivalent formulation of the pairwise distances
            pairwise_dot = tf.matmul(x, tf.transpose(x))  # (N, N) = (N, Dim) matmul (Dim, N)
            norms = tf.diag_part(pairwise_dot)  # (N,)
            norms = tf.reshape(norms, [-1, 1])  # (N, 1)
            pairwise_distance = norms - 2 * pairwise_dot + tf.transpose(norms)  # (N, N) = (N, 1) - (N, N) + (1, N)
            # Avoids small negatives due to possible rounding errors
            pairwise_distance = tf.nn.relu(pairwise_distance)  # (N, N)
        else:
            # This option is more robust to rounding errors at the cost of storing a larger tensor
            # in memory, because it performs
            # (N, 1, Dim) - (1, N, Dim) = (N, N, Dim)
            # which is the straightforward difference matrix that is then squared and reduced to (N, N)
            pairwise_difference = x[:, tf.newaxis, :] - x[tf.newaxis, :, :]  # (N, N, Dim) = (N, 1, Dim) - (1, N, Dim)
            pairwise_squared_difference = tf.square(pairwise_difference)  # (N, N, Dim)
            pairwise_distance = tf.reduce_sum(pairwise_squared_difference, axis=2)  # (N, N)

        # We don't bother with the normalization constant of the gaussian kernel
        # since it is canceled out during normalization of the Gram matrix
        den = 2 * (sigma_x ** 2)
        gram = tf.exp(-pairwise_distance / den)
        # Normalize gram
        x_dims = tf.shape(x)
        n = tf.cast(x_dims[0], tf.float32)
        norm_gram = gram / n
        return norm_gram

    def entropy(
            self, x, sigma_x=None, x_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram = self.normalized_gram(x, sigma_x, x_is_image)
        entropy = self.entropy_with_gram(norm_gram)
        return entropy

    def joint_entropy(
            self, x, y, sigma_x=None, sigma_y=None,
            x_is_image=False, y_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram_a = self.normalized_gram(x, sigma_x, x_is_image)
        norm_gram_b = self.normalized_gram(y, sigma_y, y_is_image)
        joint_entropy = self.joint_entropy_with_gram(norm_gram_a, norm_gram_b)
        return joint_entropy

    def mutual_information(
            self, x, y, sigma_x=None, sigma_y=None,
            x_is_image=False, y_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram_a = self.normalized_gram(x, sigma_x, x_is_image)
        norm_gram_b = self.normalized_gram(y, sigma_y, y_is_image)
        mi_xy = self.mutual_information_with_gram(norm_gram_a, norm_gram_b)
        return mi_xy

    def entropy_with_gram(self, norm_gram):
        with tf.device('/cpu:0'):
            eigvals, _ = tf.self_adjoint_eig(norm_gram)
            # Fix possible numerical instabilities:
            # Remove small negatives
            eigvals = tf.nn.relu(eigvals)
            # Ensure eigenvalues sum 1
            eigvals = eigvals / tf.reduce_sum(eigvals)
            # Compute entropy in the specified base
            sum_term = tf.reduce_sum(eigvals ** self.alpha)
            entropy = tf.log(sum_term) / (1.0 - self.alpha)
            entropy = entropy / np.log(self.log_base)
        return entropy

    def joint_entropy_with_gram(self, norm_gram_a, norm_gram_b):
        n = tf.cast(tf.shape(norm_gram_a)[0], tf.float32)
        norm_gram = n * tf.multiply(norm_gram_a, norm_gram_b)
        joint_entropy = self.entropy_with_gram(norm_gram)
        return joint_entropy

    def mutual_information_with_gram(self, norm_gram_a, norm_gram_b):
        h_x = self.entropy_with_gram(norm_gram_a)
        h_y = self.entropy_with_gram(norm_gram_b)
        h_xy = self.joint_entropy_with_gram(norm_gram_a, norm_gram_b)
        mi_xy = h_x + h_y - h_xy
        return mi_xy
