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
from scipy.stats import pearsonr, ttest_ind
import argparse
from pathlib import Path

from tensorly.metrics import congruence_coefficient

from independent_vector_analysis.visualization import calculate_corrcoef


def calculate_p_values_iva(mixing_matrices):
    p_values = np.zeros((mixing_matrices.shape[1], mixing_matrices.shape[2]))
    for scv_id in range(mixing_matrices.shape[1]):
        for dataset_id in range(mixing_matrices.shape[2]):
            t, p = ttest_ind(mixing_matrices[0:150, scv_id, dataset_id],
                             mixing_matrices[150:, scv_id, dataset_id], equal_var=False)
            p_values[scv_id, dataset_id] = p
    return p_values


def calculate_p_values_parafac2(subjects):
    p_values = np.zeros(subjects.shape[1])
    for comp_id in range(subjects.shape[1]):
        t, p = ttest_ind(subjects[0:150, comp_id],
                         subjects[150:, comp_id], equal_var=False)
        p_values[comp_id] = p
    return p_values


def calculate_frobenius_norm(true_cov, est_cov):
    # true_cov and est_cov both have dimensions n_datasets x n_datasets x n_components
    # return frobenius norm of difference for each component
    return np.linalg.norm(np.abs(true_cov) - np.abs(est_cov), 'fro', axis=(0, 1)) / np.linalg.norm(
        true_cov, 'fro', axis=(0, 1))


def calculate_source_correlation(true_sources, estimated_sources):
    # sources are of dimensions n_components x n_voxels x n_datasets
    # return correlation for each component in each dataset
    C, T, K = true_sources.shape
    corr = np.zeros((C, K))
    for c in range(C):
        for k in range(K):
            corr[c, k] = pearsonr(true_sources[c, :, k], estimated_sources[c, :, k])[0]
    return corr


def calculate_performance_metrics(cov_true, cov_est, sources_true, sources_est, p_values_true,
                                  p_values_est):
    # difference of true and estimated covariance matrices
    frob = calculate_frobenius_norm(cov_true, cov_est)

    # correlation of true and estimated sources
    corr = calculate_source_correlation(sources_true, sources_est)

    # accuracy of true and estimated significant p-values
    significant_true = p_values_true < 0.05
    significant_est = p_values_est < 0.05
    acc = np.sum(significant_est == significant_true, axis=1) / significant_true.shape[1]

    return frob, corr, acc


if __name__ == '__main__':
    # read arguments of terminal
    parser = argparse.ArgumentParser(description='Print args',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--scenario', type=int, help='Which data was simulated: 1 or 2.')
    parser.add_argument('--nc', type=int, default=4, help='Number of components to estimate.')
    parser.add_argument('--nmontecarlo', type=int, default=50,
                        help='Number of independent simulations.')

    args = parser.parse_args()
    # for running the code locally without using the terminal, comment the line above and
    # uncomment the following line
    # args = parser.parse_args('--scenario 1'.split())

    scenario = args.scenario
    n_c = args.nc
    n_montecarlo = args.nmontecarlo

    p_true = []

    frob_iva = []
    corr_iva = []
    acc_iva = []
    t_iva = []
    p_iva = []

    frob_parafac2 = []
    corr_parafac2 = []
    acc_parafac2 = []
    t_parafac2 = []
    p_parafac2 = []

    print('Start calculation of performance metrics...')
    for run in range(n_montecarlo):
        data = np.load(Path(Path(__file__).parent.parent,
                            f'simulations/scenario_{scenario}_run{run}.npy'),
                       allow_pickle=True).item()
        true_data = data['true']
        parafac2_results = data['parafac2']
        iva_results = data['iva']

        cov_true = true_data['scv_cov']
        mixing_true = true_data['Ak']
        sources_true = true_data['S']
        p_values_true = calculate_p_values_iva(mixing_true)

        # do not evaluate the current run if p-values are not significant for the first two
        # components
        if p_values_true[0, 0] >= 0.05 or p_values_true[2, 0] >= 0.05:
            continue

        p_true.append(p_values_true)

        # iva

        cov_iva = iva_results['scv_cov']
        mixing_iva = iva_results['A']
        sources_iva = iva_results['S']
        time_iva = iva_results['time']
        p_values_iva = calculate_p_values_iva(mixing_iva)
        p_iva.append(p_values_iva)

        frob, corr, acc = calculate_performance_metrics(
            cov_true, cov_iva, sources_true, sources_iva, p_values_true, p_values_iva)
        frob_iva.append(frob)
        corr_iva.append(corr)
        acc_iva.append(acc)
        t_iva.append(time_iva)

        # parafac2

        result = parafac2_results['result']
        time_parafac2 = parafac2_results['time']
        datasets, B, subjects = result.factors
        projections = result.projections

        # calculate SCV covariance matrices using voxels
        voxels = np.zeros((len(projections), *projections[0].shape))
        for idx, P in enumerate(projections):
            voxels[idx, :, :] = P @ B
        cov_parafac2 = calculate_corrcoef(voxels.T)

        # sort PARAFAC2 components using FMS / Tucker congruence_coefficient on voxels mode
        sources_temp = np.moveaxis(sources_true, [0, 1, 2], [2, 1, 0])
        Btilde_true = np.reshape(sources_temp, (
            sources_temp.shape[0] * sources_temp.shape[1], sources_temp.shape[2]))
        Btilde_parafac2 = np.reshape(voxels, (voxels.shape[0] * voxels.shape[1], voxels.shape[2]))
        _, permutation = congruence_coefficient(Btilde_true, Btilde_parafac2)

        voxels = voxels[:, :, permutation]
        cov_parafac2 = cov_parafac2[:, :, permutation]
        datasets = datasets[:, permutation]
        subjects = subjects[:, permutation]

        p_values_parafac2 = calculate_p_values_parafac2(subjects)
        p_parafac2.append(p_values_parafac2)
        # broadcast p values such that they can be used in the same way as IVA p values
        p_values_parafac2 = np.tile(p_values_parafac2[:, np.newaxis], p_values_true.shape[1])

        # performance metrics
        frob, corr, acc = calculate_performance_metrics(
            cov_true, cov_parafac2, sources_true, voxels.T, p_values_true, p_values_parafac2)
        frob_parafac2.append(frob)
        corr_parafac2.append(corr)
        acc_parafac2.append(acc)
        t_parafac2.append(time_parafac2)

    print(f'Save metrics as simulations/metrics_scenario_{scenario}.npy.')
    np.save(Path(Path(__file__).parent.parent,
                 f'simulations/metrics_scenario_{scenario}.npy'),
            {'iva': {'frob': frob_iva, 'corr': corr_iva, 'acc': acc_iva, 'time': t_iva,
                     'pvalues': p_iva},
             'parafac2': {'frob': frob_parafac2, 'corr': corr_parafac2, 'acc': acc_parafac2,
                          'time': t_parafac2, 'pvalues': p_parafac2},
             'true': {'pvalues': p_true}
             })
