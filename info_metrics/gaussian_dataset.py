"""
@author Nicolas Tapia
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_shannon_mut_info(dimension, corr_factor):
    # Theoretical Mutual Information
    # mut_info = -(1/2) * np.log(np.linalg.det(joint_covariance))
    mut_info = -(dimension / 2) * np.log(1 - corr_factor ** 2)
    return mut_info


def generate_dataset(dimension, corr_factor, n_samples):
    """A more efficient implementation of correlated gaussian dataset."""
    # Sample standard normal
    joint_var = np.random.rand(n_samples, 2 * dimension)
    # Transform with correlations
    var_1 = joint_var[:, :dimension]
    var_2 = joint_var[:, dimension:]
    term_1 = np.sqrt(1 + corr_factor) * var_1 / np.sqrt(2)
    term_2 = np.sqrt(1 - corr_factor) * var_2 / np.sqrt(2)
    samples_x = term_1 + term_2
    samples_z = term_1 - term_2
    return samples_x, samples_z


def generate_batch_tf(dimension, corr_factor, n_samples):
    """This is a TF implementation to sample two multivariate standard
    gaussians with per-dim correlation of the same value.
    Efficiency is increased compared to the previous implementation
    as we avoid the use of the (potentially too large) covariance matrix,
    since for this simple case, a short transformation from two standard
    gaussians is enough.
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
