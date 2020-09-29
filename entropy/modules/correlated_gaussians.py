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


class CorrelatedGaussians(object):
    """
    Implementation of a dataset composed of two standard gaussian variables
    X and Z, of the same dimensionality, with dimension-wise correlation factor
    of the same value. That is:
        corr(Xi, Zj) = corr_factor * delta_ij
    """

    def __init__(self, mode="numpy"):
        """
        Args:
             mode: (["numpy", "tensorflow"]) Mode of operation
                for sampling the dataset. If "numpy", then the dataset
                samples numpy arrays. If "tensorflow", then the dataset
                returns tensorflow operations for sampling. Defaults to
                "numpy".
        """
        if mode not in ["numpy", "tensorflow"]:
            msg = 'Expected ["numpy", "tensorflow"] for mode, but %s was provided.' % mode
            raise ValueError(msg)
        self.mode = mode

    def sample(self, dimension, corr_factor, n_samples):
        """Generates samples of correlated gaussians.

        Args:
            dimension: (int) Dimensionality of each variable.
            corr_factor: (float or tensor, depending of mode) Dimension-wise
                correlation factor.
            n_samples: (int or tensor, depending of mode) Number of samples.

        Returns:
             samples_x: Samples of first variable.
             samples_z: Samples of second variable.
        """
        if self.mode == "tensorflow":
            samples_x, samples_z = self._generate_batch_tf(
                dimension, corr_factor, n_samples)
        else:
            samples_x, samples_z = self._generate_batch_np(
                dimension, corr_factor, n_samples)
        return samples_x, samples_z

    @staticmethod
    def get_shannon_mi(dimension, corr_factor):
        """Computes the theoretical mutual information of the pair of variables.

        Args:
            dimension: (int) Dimensionality of each variable.
            corr_factor: (float) Dimension-wise correlation factor.
        """
        # Theoretical Mutual Information for standard gaussians:
        # mut_info = -(1/2) * np.log(np.linalg.det(joint_covariance))
        # For the correlated gaussians dataset, the determinant of the
        # joint covariance has a simpler expression:
        mut_info = -(dimension / 2) * np.log(1 - corr_factor ** 2)
        return mut_info

    @staticmethod
    def _generate_batch_np(dimension, corr_factor, n_samples):
        """Generates samples of numpy arrays.
        See 'sample' method for a description.

        This is an efficient implementation that avoids the use of the
        (potentially too large) covariance matrix, since for this simple case,
        a short transformation from two independent standard gaussians is
        enough.
        """
        # Sample standard normal
        joint_var = np.random.normal(size=(n_samples, 2 * dimension))
        # Transform with correlations
        var_1 = joint_var[:, :dimension]
        var_2 = joint_var[:, dimension:]
        term_1 = np.sqrt(1 + corr_factor) * var_1 / np.sqrt(2)
        term_2 = np.sqrt(1 - corr_factor) * var_2 / np.sqrt(2)
        samples_x = term_1 + term_2
        samples_z = term_1 - term_2
        return samples_x, samples_z

    @staticmethod
    def _generate_batch_tf(dimension, corr_factor, n_samples):
        """Generates a tensorflow operation for sampling.
        See 'sample' method for a description.

        This is an efficient implementation that avoids the use of the
        (potentially too large) covariance matrix, since for this simple case,
        a short transformation from two independent standard gaussians is
        enough.
        """
        mean = np.zeros(2 * dimension).astype(np.float32)
        std = np.ones(2 * dimension).astype(np.float32)
        # Sample standard normal
        dist = tf.distributions.Normal(loc=mean, scale=std)
        joint_var = dist.sample([n_samples])
        # Transform with correlations
        var_1 = joint_var[:, :dimension]
        var_2 = joint_var[:, dimension:]
        term_1 = tf.sqrt(1 + corr_factor) * var_1 / np.sqrt(2)
        term_2 = tf.sqrt(1 - corr_factor) * var_2 / np.sqrt(2)
        samples_x = term_1 + term_2
        samples_z = term_1 - term_2
        return samples_x, samples_z
