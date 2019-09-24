from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
sns.set(style="whitegrid")
import matplotlib
label_size = 9
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
import matplotlib.pyplot as plt

# Project root path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from entropy.modules.correlated_gaussians import CorrelatedGaussians
from entropy.modules.matrix_estimator import MatrixEstimator
from entropy.utils import constants

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'entropy', 'results')


if __name__ == '__main__':
    # Settings of experiment
    batch_size_list = [100]
    sigma_zero_list = [1.0]
    dimension_list = [1, 10, 100, 1000]
    n_tries = 10
    normalize_scale = True
    normalize_dimension = True
    test_name = 'matrix'

    data = CorrelatedGaussians(mode=constants.MODE_TENSORFLOW)

    print('Correlated Gaussians Test for %s:' % test_name)
    print('Batch size list', batch_size_list)
    print('Sigma zero list', sigma_zero_list)
    print('Dimension list', dimension_list)
    print('')

    n_dims = len(dimension_list)
    figsize = (3 * n_dims, 6)

    n_dims = len(dimension_list)
    corr_factors = [i * 0.1 for i in range(10)]
    corr_neg = [-corr for corr in corr_factors][::-1]
    corr_factor_list = corr_neg[0:-1] + corr_factors

    # Shannon mutual information theoretical value
    print('Computing Shannon Values')
    shannon_data = {'dimension': [], 'corr_factor': [], 'value': []}
    for i, dimension in enumerate(dimension_list):
        print('Dimension %d' % dimension)
        for corr_factor in corr_factor_list:
            mut_info = data.get_shannon_mi(
                dimension, corr_factor)
            shannon_data['dimension'].append(dimension)
            shannon_data['corr_factor'].append(corr_factor)
            shannon_data['value'].append(mut_info)

    # Estimation
    start_time = time.time()
    for batch_size in batch_size_list:
        # Saving data
        estimator_data = {'dimension': [], 'corr_factor': [], 'sigma_0': [],
                          'try': [], 'value': []}
        for sigma_zero in sigma_zero_list:
            estimator = MatrixEstimator(
                sigma_zero,
                normalize_scale=normalize_scale,
                normalize_dimension=normalize_dimension)

            for dimension in dimension_list:
                print('\nSamples %d. Sigma_0 %s. Dimension %d'
                      % (batch_size, sigma_zero, dimension))

                tf.reset_default_graph()
                corr_factor_ph = tf.placeholder(tf.float32, shape=())
                x_samples, y_samples = data.sample(
                    dimension, corr_factor_ph, batch_size)
                mi_estimation_tf = estimator.mutual_information(
                    x_samples, y_samples)
                # Tensorflow session for graph management
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
        
                for corr_factor in corr_factor_list:
                    print('Correlation factor %1.1f' % corr_factor)
                    estimation_tries_list = []
                    for i_try in range(n_tries):
                        mi_estimation_np = sess.run(
                            mi_estimation_tf,
                            feed_dict={corr_factor_ph: corr_factor})
        
                        estimator_data['dimension'].append(dimension)
                        estimator_data['corr_factor'].append(corr_factor)
                        estimator_data['sigma_0'].append(sigma_zero)
                        estimator_data['try'].append(i_try)
                        estimator_data['value'].append(mi_estimation_np)
        end_time = time.time()
        print('E.T.:', end_time - start_time)

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # Plot results
        shannon_df = pd.DataFrame.from_dict(shannon_data)
        estimator_df = pd.DataFrame.from_dict(estimator_data)

        shannon_df['name'] = 'Shannon MI'
        estimator_df['name'] = 'Estimation'
        # columns: dimension  corr_factor  sigma_0  try     value

        sigma_0_list = estimator_df.sigma_0.unique()
        dimension_list = estimator_df.dimension.unique()
        n_dims = len(dimension_list)
    
        for s in sigma_0_list:
            print('Sigma', s)
            fig, ax = plt.subplots(
                2, n_dims, figsize=figsize, sharex=True, dpi=100)
    
            for j, d in enumerate(dimension_list):
                mini_shannon_df = shannon_df.loc[
                    (shannon_df['dimension'] == d)]
                ax[0, j] = sns.lineplot(
                    x="corr_factor", y="value", style='name', markers=True,
                    data=mini_shannon_df, ax=ax[0, j])
                handles, labels = ax[0, j].get_legend_handles_labels()
                ax[0, j].legend(handles=handles[1:], labels=labels[1:])
                # ax[0, j].legend(loc='upper center')
                ax[0, j].set_ylabel('')
                ax[0, j].set_title('Dimensionality: %d' % d)
    
                mini_estimator_df = estimator_df.loc[
                    (estimator_df['dimension'] == d)
                    & (estimator_df['sigma_0'] == s)]
                ax[1, j] = sns.lineplot(
                    x="corr_factor", y="value", style='name', markers=True,
                    data=mini_estimator_df, ax=ax[1, j])
                handles, labels = ax[1, j].get_legend_handles_labels()
                ax[1, j].legend(handles=handles[1:], labels=labels[1:])
                ax[1, j].set_ylabel('')
                ax[1, j].set_xlabel('Correlation Factor')
                ax[1, j].set_title(
                    'Batch size: %d, $\sigma_0$: %1.1f' % (batch_size, s))
                ax[1, j].yaxis.set_major_formatter(y_formatter)
            plt.tight_layout()

            os.makedirs(RESULTS_DIR, exist_ok=True)
            filename = os.path.join(
                RESULTS_DIR,
                'gaussian_%s_batch%d.png' % (test_name, batch_size))
            fig.savefig(filename)
