from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time

import numpy as np
import tensorflow as tf
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib
label_size = 9
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
import matplotlib.pyplot as plt
import pandas as pd

from info_metrics import gaussian_dataset
from info_metrics.information_estimator_v2 import InformationEstimator

y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)


if __name__ == '__main__':

    start_time = time.time()

    # Settings of experiment
    batch_size_list = [100]
    sigma_zero_list = [2.0]
    n_tries = 10
    dimension_list = [1, 10, 100, 1000, 5000]

    print('Batch size list', batch_size_list)
    print('Sigma zero list', sigma_zero_list)
    print('Dimension list', dimension_list)
    print('')

    figsize = (15, 6)

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
            mut_info = gaussian_dataset.get_shannon_mut_info(
                dimension, corr_factor)
            shannon_data['dimension'].append(dimension)
            shannon_data['corr_factor'].append(corr_factor)
            shannon_data['value'].append(mut_info)

    # Estimation
    for batch_size in batch_size_list:
        # Saving data
        principe_data = {'dimension': [], 'corr_factor': [], 'sigma_0': [],
                         'try': [], 'value': []}

        for sigma_zero in sigma_zero_list:
            for dimension in dimension_list:
                print('\nSamples %d, Sigma_0 %s' % (batch_size, sigma_zero))
                print('Dimension %d' % dimension)
        
                # Estimation
                tf.reset_default_graph()
                estimator = InformationEstimator(sigma_zero)
                corr_factor_ph = tf.placeholder(tf.float32, shape=())
                x_samples, y_samples = gaussian_dataset.generate_batch_tf(
                    dimension, corr_factor_ph, batch_size)
                mi_estimation_tf = estimator.mutual_information(x_samples, y_samples)
                sess = tf.Session()
        
                for corr_factor in corr_factor_list:
                    print('Correlation factor %1.1f' % corr_factor)
                    estimation_tries_list = []
                    for i_try in range(n_tries):
                        mi_estimation_np = sess.run(
                            mi_estimation_tf,
                            feed_dict={corr_factor_ph: corr_factor})
        
                        principe_data['dimension'].append(dimension)
                        principe_data['corr_factor'].append(corr_factor)
                        principe_data['sigma_0'].append(sigma_zero)
                        principe_data['try'].append(i_try)
                        principe_data['value'].append(float(mi_estimation_np))     

        end_time = time.time()
        print('E.T.:', end_time - start_time)

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # Plot results
        shannon_df = pd.DataFrame.from_dict(shannon_data)
        principe_df = pd.DataFrame.from_dict(principe_data)

        shannon_df['name'] = 'Shannon MI'
        principe_df['name'] = 'Estimation'
        # columns: dimension  corr_factor  sigma_0  try     value

        sigma_0_list = principe_df.sigma_0.unique()
        dimension_list = principe_df.dimension.unique()
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
    
                mini_principe_df = principe_df.loc[
                    (principe_df['dimension'] == d)
                    & (principe_df['sigma_0'] == s)]
                ax[1, j] = sns.lineplot(
                    x="corr_factor", y="value", style='name', markers=True,
                    data=mini_principe_df, ax=ax[1, j])
                handles, labels = ax[1, j].get_legend_handles_labels()
                ax[1, j].legend(handles=handles[1:], labels=labels[1:])
                ax[1, j].set_ylabel('')
                ax[1, j].set_xlabel('Correlation Factor')
                ax[1, j].set_title(
                    'Batch size: %d, $\sigma_0$: %1.1f' % (batch_size, s))
                ax[1, j].yaxis.set_major_formatter(y_formatter)
            plt.tight_layout()
            filename = 'gaussian_batch%d_sigma%1.1f.png' % (batch_size, s)
            fig.savefig(filename)
