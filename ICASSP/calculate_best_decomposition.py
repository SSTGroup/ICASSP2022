#     Copyright (c) <2022> <University of Paderborn>
#     Signal and System Theory Group, Univ. of Paderborn, https://sst-group.org/
#     https://github.com/SSTGroup/ICASSP2022
#
#     Permission is hereby granted, free of charge, to any person
#     obtaining a copy of this software and associated documentation
#     files (the "Software"), to deal in the Software without restriction,
#     including without limitation the rights to use, copy, modify and
#     merge the Software, subject to the following conditions:
#
#     1.) The Software is used for non-commercial research and
#        education purposes.
#
#     2.) The above copyright notice and this permission notice shall be
#        included in all copies or substantial portions of the Software.
#
#     3.) Publication, Distribution, Sublicensing, and/or Selling of
#        copies or parts of the Software requires special agreements
#        with the University of Paderborn and is in general not permitted.
#
#     4.) Modifications or contributions to the software must be
#        published under this license. The University of Paderborn
#        is granted the non-exclusive right to publish modifications
#        or contributions in future versions of the Software free of charge.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#     OTHER DEALINGS IN THE SOFTWARE.
#
#     Persons using the Software are encouraged to notify the
#     Signal and System Theory Group at the University of Paderborn
#     about bugs. Please reference the Software in your publications
#     if it was used for them.


import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.random

# parallel processing
import multiprocessing
from joblib import Parallel, delayed

from tqdm import tqdm


def nn_parafac2(X_, n_c):
    # random initialization factor matrices with B as identity
    shapes = [Xk.shape for Xk in X_]
    init = tl.random.random_parafac2(shapes, n_c)
    init.factors[1] = np.eye(n_c)
    # run PARAFAC2 decomposition with non-negativity in the first mode
    result, error = tl.decomposition.parafac2(X_, n_c, nn_modes=[0], init=init,
                                              normalize_factors=True, tol=1e-5,
                                              return_errors=True, n_iter_max=5000)
    return result, error


def best_run(X, n_c, n_runs):
    """
    Run PARAFAC2 tensor decomposition with non-negativity in the tasks/datasets mode several times
    with random init plus one time with svd init.
    Return run which achieves most consistent error.

    Parameters
    ----------
    X : np.ndarray
        tensor to be decomposed

    n_c : int
        number of components in which tensor is decomposed

    n_runs : int
        number of times that parafac2 is performed with random init


    Returns
    -------
    result : tl.cp_tensor
        decomposition of best run

    error_change : list
        list with errors of all iterations for the best run

    initialization_errors: list
        minimum reconstruction error for each random + svd initialization

    """

    # permute dimensions such that dimension is: tasks/datasets x voxels x subjects
    X_ = np.moveaxis(X, [0, 1, 2], [2, 1, 0])

    # random inits
    results, errors = zip(*Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(nn_parafac2)(X_, n_c) for i in tqdm(range(n_runs))))
    results = list(results)
    errors = list(errors)

    # svd init
    result, error = tl.decomposition.parafac2(X_, n_c, nn_modes=[0], init='svd',
                                              normalize_factors=True, tol=1e-5,
                                              return_errors=True, n_iter_max=5000)
    results.append(result)
    errors.append(error)

    initialization_errors = [e[-1] for e in errors]
    best_run = np.argmin(initialization_errors)
    error_change = errors[best_run]

    return results[best_run], error_change, initialization_errors
