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

from independent_vector_analysis.helpers_iva import whiten_data
from independent_vector_analysis import consistent_iva


def best_run(X, n_components, which_iva='iva_g', **kwargs):
    """
    Perform PCA on X to reduce its dimension.
    Apply IVA several times to dimension reduces dataset and return the most consistent demixing
    matrix with the corresponding sources, mixing matrix, SCV covariance matrices, and optionally
    cross joint isi.


    Parameters
    ----------
    X : np.ndarray
        real-valued data matrix of dimensions N x T x K.
        Data observations are from K data sets, where N is the number of subjects or time points
        and T is the number of samples (voxels).

    n_components : int
        PCA order

    which_iva : str
        'iva_g' or 'iva_l_sos'

    kwargs : list
        keyword arguments for the consistent_iva function


    Returns
    -------
    iva_results : dict
        - 'W' : estimated demixing matrix of dimensions n_components x n_components x K
        - 'W_change' : change in W for each iteration
        - 'S' : estimated sources of dimensions n_components x T x K
        - 'A' : estimated mixing matrix of dimensions N x n_components x K
        - 'scv_cov' : covariance matrices of the SCVs, of dimensions K x K x N
        - 'cross_isi' : cross joint isi for each run

    """

    N, T, K = X.shape

    # Dimension reduction
    X_reduced, V = whiten_data(X, n_components)

    if 'A' in kwargs.keys():
        kwargs['A'] = np.einsum('nNk, Nvk-> nvk', V, kwargs['A'])

    iva_results = consistent_iva(X=X_reduced, which_iva=which_iva, **kwargs)

    # update mixing matrix (such that dimension fits)
    A_hat = np.zeros((N, n_components, K))
    for k in range(K):
        A_hat[:, :, k] = np.linalg.lstsq(iva_results['S'][:, :, k].T, X[:, :, k].T)[0].T
    iva_results['A'] = A_hat

    return iva_results
