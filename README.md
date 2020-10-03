# entropy

Entropy and Mutual Information Estimation.

A TensorFlow implementation of the matrix-based mutual information estimator originally proposed in:

Giraldo, L.G.S., Rao, M., & Principe, J.C. (2015). Measures of entropy from data using infinitely divisible kernels. IEEE Transactions on Information Theory, 61(1), 535-548.

This estimator uses a kernel, typically chosen to be the gaussian kernel that depends on a parameter called kernel width. In our implementation, we use a modified kernel width selection rule, proposed as part of our work on information plane analysis:

N. I. Tapia and P. A. Estévez, "On the Information Plane of Autoencoders," *2020 International Joint Conference on Neural Networks (IJCNN)*, Glasgow, United Kingdom, 2020, pp. 1-8, doi: 10.1109/IJCNN48605.2020.9207269.

Full text available at: https://arxiv.org/abs/2005.07783

Please see the demo Jupyter Notebook in "scripts" for usage demonstration. This implementation was made before TensorFlow 2, so you need TensorFlow 1 (i.e. tf 1.x). If you find this software useful, please consider citing our work.

