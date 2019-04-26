"""Module that contains several useful checks for the project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def check_array(x, ndim=2):
    """Ensures that x is a numpy array, and raises a ValueError exception if
    x is not ndim-dimensional."""
    if type(x) is not np.ndarray:
        x = np.asarray(x)
    if x.ndim != ndim:
        raise ValueError('Only %d-dimensional arrays are accepted' % ndim)


def check_valid_value(value, name, valid_list):
    """Raises a ValueError exception if value not in valid_list."""
    if value not in valid_list:
        msg = 'Expected %s for %s, but %s was provided.' \
              % (valid_list, name, value)
        raise ValueError(msg)


def is_integer(x):
    """Checks if the array x contains only integers."""
    return np.array_equal(x, x.astype(np.int32))
