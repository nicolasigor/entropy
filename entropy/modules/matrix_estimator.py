"""
@author Nicolas Tapia
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from entropy.utils import checks
from entropy.utils import constants

EPS = np.finfo(float).eps


class MatrixEstimator(object):
    """ TensorFlow implementation of estimator proposed in:

    Giraldo, L. G. S., Rao, M., & Principe, J. C. (2015). Measures of entropy
    from data using infinitely divisible kernels. IEEE Transactions on
    Information Theory, 61(1), 535-548.

    with modifications made by the authors to attenuate scale and dimension
    related artifacts. The originally proposed kernel size formula is:

    sigma = sigma_0 * n ^ (-1 / (4+d))

    If normalize_scale is set to True, then the variable is first normalized
    to zero mean and unit variance, as:

    x -> (x - mean) / sqrt(var + epsilon)

    which is equivalent to add the standard deviation as a multiplicative
    dependence in sigma, as done in the classical Silverman rule. This is done
    to achieve invariance to changes of scale during the mutual information
    estimation process. Epsilon is a small number to avoid division by zero.

    If normalize_dimension is set to True, then sigma is computed as:

    sigma = sigma_0 * sqrt(d) * n ^ (-1 / (4+d))

    This is done to center the distribution of pair-wise distances to the same
    mean across variables with different dimensions, and as a consequence
    to attenuate dimension related artifacts.

    Note that normalize_scale=False and normalize_dimension=False will give you
    the original version of the estimator.
    """

    def __init__(
            self,
            sigma_zero,
            alpha=1.01,
            epsilon=1e-8,
            normalize_scale=True,
            normalize_dimension=True
    ):
        self.sigma_zero = sigma_zero
        self.alpha = alpha
        self.epsilon = epsilon
        self.normalize_scale = normalize_scale
        self.normalize_dimension = normalize_dimension

    def _compute_sigma(self, x):
        x_dims = tf.shape(x)
        n = tf.to_float(x_dims[0])
        d = tf.to_float(x_dims[1])
        sigma = self.sigma_zero * n ** (-1 / (4 + d))
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
        """If sigma_x is provided, then that value will be used. Otherwise
        it will be automatically computed using the formula.
        If x_is_image is True, then the normalization of scale (if applicable)
        is done mixing dimensions. If false, each dimension is normalized
        independently.
        """
        if sigma_x is None:
            sigma_x = self._compute_sigma(x)
        if self.normalize_scale:
            x = self._normalize_variable(x, x_is_image)

        # # Compute differences directly to avoid numerical instability
        # pairwise_difference = x[:, tf.newaxis, :] - x[tf.newaxis, :, :]
        # pairwise_squared_difference = tf.square(pairwise_difference)
        # pairwise_distance = tf.reduce_sum(
        #     pairwise_squared_difference, axis=2)

        pairwise_dot = tf.matmul(x, tf.transpose(x))
        norms = tf.diag_part(pairwise_dot)
        norms = tf.reshape(norms, [-1, 1])
        pairwise_distance = norms - 2 * pairwise_dot + tf.transpose(norms)

        # Avoids small negatives due to numerical precision
        pairwise_distance = tf.nn.relu(pairwise_distance)
        # We don't bother with the normalization constant of the kernel
        # since it will be canceled during normalization of the Gram matrix
        den = 2 * (sigma_x ** 2)
        gram = tf.exp(-pairwise_distance / den)
        # Normalize gram
        x_dims = tf.shape(x)
        n = tf.to_float(x_dims[0])
        norm_gram = gram / n
        return norm_gram

    def entropy(self, x, sigma_x=None, x_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram = self.normalized_gram(x, sigma_x, x_is_image)
        entropy = self.entropy_with_gram(norm_gram)
        return entropy

    def joint_entropy(self, x, y, sigma_x=None, sigma_y=None,
                      x_is_image=False, y_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram_a = self.normalized_gram(x, sigma_x, x_is_image)
        norm_gram_b = self.normalized_gram(y, sigma_y, y_is_image)
        joint_entropy = self.joint_entropy_with_gram(norm_gram_a, norm_gram_b)
        return joint_entropy

    def mutual_information(self, x, y, sigma_x=None, sigma_y=None,
                           x_is_image=False, y_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram_a = self.normalized_gram(x, sigma_x, x_is_image)
        norm_gram_b = self.normalized_gram(y, sigma_y, y_is_image)
        mi_xy = self.mutual_information_with_gram(norm_gram_a, norm_gram_b)
        return mi_xy

    def conditional_entropy(self, x, y):
        pass

    def conditional_mutual_information(self, x, y, z):
        pass

    def entropy_with_gram(self, norm_gram):
        with tf.device('/cpu:0'):
            eigvals, _ = tf.self_adjoint_eig(norm_gram)
            eigvals = tf.nn.relu(eigvals)  # Avoids small negatives

            # Ensure eigenvalues sum 1,
            # in case a numerical instability occurred.
            eigvals = eigvals / tf.reduce_sum(eigvals)

            sum_term = tf.reduce_sum(eigvals ** self.alpha)
            entropy = tf.log(sum_term) / (1.0 - self.alpha)
        return entropy

    def joint_entropy_with_gram(self, norm_gram_a, norm_gram_b):
        n = tf.to_float(tf.shape(norm_gram_a)[0])
        norm_gram = n * tf.multiply(norm_gram_a, norm_gram_b)
        joint_entropy = self.entropy_with_gram(norm_gram)
        return joint_entropy

    def mutual_information_with_gram(self, norm_gram_a, norm_gram_b):
        h_x = self.entropy_with_gram(norm_gram_a)
        h_y = self.entropy_with_gram(norm_gram_b)
        h_xy = self.joint_entropy_with_gram(norm_gram_a, norm_gram_b)
        mi_xy = h_x + h_y - h_xy
        return mi_xy

    def conditional_entropy_with_gram(self, norm_gram_a, norm_gram_b):
        pass

    def conditional_mutual_information_with_gram(
            self, norm_gram_a, norm_gram_b, norm_gram_c):
        pass
