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
import argparse
from pathlib import Path
import time

from independent_vector_analysis.data_generation import MGGD_generation
from ICASSP import calculate_best_iva
from ICASSP import calculate_best_decomposition


def simulate_scenario_1(cov, N, T, K, sigma_n, step):
    n_c = cov.shape[-1]  # components

    # generate sources according to each SCV
    sources = np.zeros((n_c, T, K))
    for idx in range(n_c):
        sources[idx, :, :] = MGGD_generation(T, Sigma=cov[:, :, idx])[0].T

    # create mixing matrices
    A = np.zeros((N, n_c))

    # component 0 and 2 are different in all datasets
    A[:, [0, 2]] = np.random.randn(N, 2) * sigma_n
    A[0:N // 2, [0, 2]] += step

    # components 1 and 3 have same variance as other two
    sigma_a = 1 / 2 * (np.std(A[:, 0]) + np.std(A[:, 2]))
    A[:, [1, 3]] = np.random.randn(N, n_c - 2) * sigma_a

    C = 1.5 + np.random.randn(K, n_c) * 0.1
    C[4:, 2] -= 1
    C[0:4, 1] -= 1
    C[:, 3] = 1.5 + np.random.randn(K) * 0.5  # so that first and last column of C are not similar

    # A^[k] = A @ diag(c^[k])
    mixing = np.zeros((N, n_c, K))
    for k in range(K):
        mixing[:, :, k] = A @ np.diag(C[k, :])

    observations = np.zeros((N, T, K))
    for k in range(K):
        observations[:, :, k] = mixing[:, :, k] @ sources[:, :, k]

    return mixing, sources, observations, A, C


def simulate_scenario_2(cov, N, T, K, sigma_n, step):
    n_c = cov.shape[-1]  # components

    # generate sources according to each SCV
    sources = np.zeros((n_c, T, K))
    for idx in range(n_c):
        sources[idx, :, :] = MGGD_generation(T, Sigma=cov[:, :, idx])[0].T

    # create mixing matrices
    A = np.zeros((N, n_c))

    # component 0 is different in all datasets
    A[:, 0] = np.random.randn(N) * sigma_n
    A[0:N // 2, 0] += step

    # other components have same variance as first
    sigma_a = np.std(A[:, 0])
    A[:, 1:] = np.random.randn(N, n_c - 1) * sigma_a

    C = 1.5 + np.random.randn(K, n_c) * 0.1
    C[4:, 2] -= 1
    C[0:4, 1] -= 1
    C[:, 3] = 1.5 + np.random.randn(K) * 0.5  # so that first and last column of C are not similar

    # A^[k] = A @ diag(c^[k])
    mixing = np.zeros((N, n_c, K))
    for k in range(K):
        mixing[:, :, k] = A @ np.diag(C[k, :])

    # component 2 is different in first 4 datasets, but should have same variance as component 0
    A2 = np.copy(A)
    A2[:, 2] = A2[:, 2] * sigma_n / sigma_a
    A2[0:N // 2, 2] += step
    for k in range(4):
        # same as:  mixing[:, 1, k] = A1 @ np.diag(C[k, :])[1, :]
        mixing[:, :, k] = A2 @ np.diag(C[k, :])

    observations = np.zeros((N, T, K))
    for k in range(K):
        observations[:, :, k] = mixing[:, :, k] @ sources[:, :, k]

    return mixing, sources, observations, [A, A2], C


if __name__ == '__main__':
    # read arguments of terminal
    parser = argparse.ArgumentParser(description='Print args',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--scenario', type=int, help='Which scenario to simulate: 1 or 2.')
    parser.add_argument('--nc', type=int, default=4, help='Number of components to estimate.')
    parser.add_argument('--niva', type=int, default=500,
                        help='Number of runs to pick best IVA result.')
    parser.add_argument('--nparafac2', type=int, default=50,
                        help='Number of runs to pick best PARAFAC2 result.')
    parser.add_argument('--nmontecarlo', type=int, default=50,
                        help='Number of independent simulations.')

    args = parser.parse_args()
    # for running the code locally without using the terminal, comment the line above and
    # uncomment the following line
    # args = parser.parse_args('--scenario 1'.split())

    scenario = args.scenario
    n_c = args.nc
    n_runs_iva = args.niva
    n_runs_parafac2 = args.nparafac2
    n_montecarlo = args.nmontecarlo

    print(f'Starting simulation of scenario {scenario} with {n_c} components.')

    cov_true = np.load(Path(Path(__file__).parent.parent,
                            'simulations/simulated_cov_with_noise.npy'))
    K = cov_true.shape[0]  # datasets
    nc = cov_true.shape[2]  # components
    N = 300  # subjects
    T = 5000  # voxels
    sigma_n = 1
    step = 0.5

    for run in range(n_montecarlo):
        print(f'Start run {run}...')
        if scenario == 1:
            mixing_true, sources_true, observations, A_true, C_true = simulate_scenario_1(
                cov_true, N, T, K, sigma_n, step)
        elif scenario == 2:
            mixing_true, sources_true, observations, A_true, C_true = simulate_scenario_2(
                cov_true, N, T, K, sigma_n, step)
        else:
            raise ValueError(f"'scenario' must be 1 or 2.")

        true_data = {'scv_cov': cov_true, 'S': sources_true, 'Ak': mixing_true,
                     'X': observations, 'A': A_true, 'C': C_true}

        # iva
        print('Computing IVA ...')
        t_start = time.time()
        iva_results = calculate_best_iva.best_run(observations, A=mixing_true, n_components=n_c,
                                                  n_runs=n_runs_iva)
        t_end = time.time()
        iva_results['time'] = t_end - t_start

        # parafac2
        print('Computing PARAFAC2 ...')
        t_start = time.time()
        parafac2_result, error_change, init_errors = calculate_best_decomposition.best_run(
            observations, n_c, n_runs=n_runs_parafac2)
        t_end = time.time()
        parafac2_time = t_end - t_start

        print(f'Save run as simulations/scenario_{scenario}_run{run}.npy.')
        np.save(Path(Path(__file__).parent.parent,
                     f'simulations/scenario_{scenario}_run{run}.npy'),
                {'true': true_data,
                 'iva': iva_results,
                 'parafac2': {'result': parafac2_result, 'error_change': error_change,
                              'init_errors': init_errors, 'time': parafac2_time}
                 })
