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


import matplotlib.pyplot as plt
import tikzplotlib


def boxplots_scenario1(p_values_true, p_values_iva, p_values_parafac2, filename=None):
    """
    plot p-value distribution for each component in individual subplot


    Parameters
    ----------
    p_values_true: np.ndarray
        true p-values of dimension (n_runs, n_components, n_datasets)

    p_values_iva : np.ndarray
        p-values estimated by IVA of dimension (n_runs, n_components, n_datasets)

    p_values_parafac2 : np.ndarray
        p-values estimated by PARAFAC2 of dimension (n_runs, n_components)

    Returns
    -------

    """

    # number of components
    nc = p_values_true.shape[1]

    fig, axes = plt.subplots(figsize=(2 * nc, nc), nrows=1, ncols=nc)
    if filename is None:
        fig.suptitle('Distribution of p-values')

    # Set the colors for each distribution
    colors = ['C0', 'C1', 'C2']
    colors_true = dict(color=colors[0])
    colors_iva = dict(color=colors[1])
    colors_parafac2 = dict(color=colors[2])

    for idx in range(nc):
        axes[idx].boxplot(p_values_true[:, idx, 0], positions=[1 + idx * 4],
                          labels=['true'], boxprops=colors_true,
                          medianprops=colors_true, whiskerprops=colors_true,
                          capprops=colors_true, flierprops=dict(markeredgecolor=colors[0]))

        axes[idx].boxplot(p_values_iva[:, idx, :].flatten(), positions=[2 + idx * 4],
                          labels=['IVA-G'], boxprops=colors_iva,
                          medianprops=colors_iva, whiskerprops=colors_iva,
                          capprops=colors_iva, flierprops=dict(markeredgecolor=colors[1]))

        axes[idx].boxplot(p_values_parafac2[:, idx], positions=[3 + idx * 4],
                          labels=['PARAFAC2'], boxprops=colors_parafac2,
                          medianprops=colors_parafac2, whiskerprops=colors_parafac2,
                          capprops=colors_parafac2, flierprops=dict(markeredgecolor=colors[2]))

        if idx == 0 or idx == 2:
            axes[idx].set_ylim([0, 0.1])
            axes[idx].set_yticks([0, 0.05, 0.1])
            axes[idx].set_yticklabels(['$0.0$', '$0.05$', '$0.1$'])
        else:
            axes[idx].set_ylim([0, 1])
            axes[idx].set_yticks([0, 0.5, 1])
            axes[idx].set_yticklabels(['$0.0$', '$0.5$', '$1.0$'])

        axes[idx].set_xticks([1 + idx * 4, 2 + idx * 4, 3 + idx * 4])
        axes[idx].set_xticklabels(['true', 'IVA-G', 'PARAFAC2'], rotation=90)
        if filename is not None:
            axes[idx].set_title(r'\large{$\mathbf{a}^{[k]}_' + str(idx + 1) + '$}')
        else:
            axes[idx].set_title(r'$\mathbf{a}^{[k]}_' + str(idx + 1) + '$')

    plt.tight_layout()
    if filename is not None:
        tikzplotlib.save(filename, encoding='utf8', axis_width='4cm')
    plt.show()


def boxplots_scenario2(p_values_true, p_values_iva, p_values_parafac2, filename=None):
    """
    plot p-value distribution for each component in individual subplot


    Parameters
    ----------
    p_values_true: np.ndarray
        true p-values of dimension (n_runs, n_components, n_datasets)

    p_values_iva : np.ndarray
        p-values estimated by IVA of dimension (n_runs, n_components, n_datasets)

    p_values_parafac2 : np.ndarray
        p-values estimated by PARAFAC2 of dimension (n_runs, n_components)

    Returns
    -------

    """

    # number of components
    nc = p_values_true.shape[1] + 1

    fig, axes = plt.subplots(figsize=(2 * nc, nc), nrows=1, ncols=nc)

    if filename is None:
        fig.suptitle('Distribution of p-values')

    # Set the colors for each distribution
    colors = ['C0', 'C1', 'C2']
    colors_true = dict(color=colors[0])
    colors_iva = dict(color=colors[1])
    colors_parafac2 = dict(color=colors[2])

    for idx in [0, 1, 3]:
        if idx == 3:
            idx_ = idx + 1
        else:
            idx_ = idx

        axes[idx_].boxplot(p_values_true[:, idx, 0], positions=[1 + idx * 4],
                           labels=['true'], boxprops=colors_true,
                           medianprops=colors_true, whiskerprops=colors_true,
                           capprops=colors_true, flierprops=dict(markeredgecolor=colors[0]))

        axes[idx_].boxplot(p_values_iva[:, idx, :].flatten(), positions=[2 + idx * 4],
                           labels=['IVA-G'], boxprops=colors_iva,
                           medianprops=colors_iva, whiskerprops=colors_iva,
                           capprops=colors_iva, flierprops=dict(markeredgecolor=colors[1]))

        axes[idx_].boxplot(p_values_parafac2[:, idx], positions=[3 + idx * 4],
                           labels=['PARAFAC2'], boxprops=colors_parafac2,
                           medianprops=colors_parafac2, whiskerprops=colors_parafac2,
                           capprops=colors_parafac2, flierprops=dict(markeredgecolor=colors[2]))

        if idx_ == 0:
            axes[idx_].set_ylim([0, 0.1])
            axes[idx_].set_yticks([0, 0.05, 0.1])
            axes[idx_].set_yticklabels(['$0.0$', '$0.05$', '$0.1$'])
        else:
            axes[idx_].set_ylim([0, 1])
            axes[idx_].set_yticks([0, 0.5, 1])
            axes[idx_].set_yticklabels(['$0.0$', '$0.5$', '$1.0$'])
        axes[idx_].set_xticks([1 + idx * 4, 2 + idx * 4, 3 + idx * 4])
        axes[idx_].set_xticklabels(['true', 'IVA-G', 'PARAFAC2'], rotation=90)
        if filename is not None:
            axes[idx_].set_title(r'\Large{$\mathbf{a}^{[k]}_' + str(idx + 1) + '$}')
        else:
            axes[idx_].set_title(r'$\mathbf{a}^{[k]}_' + str(idx + 1) + '$')

    # Now component 3 in two boxplots
    ax_idx = 2
    axes[ax_idx].boxplot(p_values_true[:, 2, 0], positions=[1 + ax_idx * 4],
                         labels=['true'], boxprops=colors_true,
                         medianprops=colors_true, whiskerprops=colors_true,
                         capprops=colors_true, flierprops=dict(markeredgecolor=colors[0]))

    axes[ax_idx].boxplot(p_values_iva[:, 2, 0:4].flatten(), positions=[2 + ax_idx * 4],
                         labels=['IVA-G'], boxprops=colors_iva,
                         medianprops=colors_iva, whiskerprops=colors_iva,
                         capprops=colors_iva, flierprops=dict(markeredgecolor=colors[1]))

    axes[ax_idx].boxplot(p_values_parafac2[:, 2], positions=[3 + ax_idx * 4],
                         labels=['PARAFAC2'], boxprops=colors_parafac2,
                         medianprops=colors_parafac2, whiskerprops=colors_parafac2,
                         capprops=colors_parafac2, flierprops=dict(markeredgecolor=colors[2]))

    axes[ax_idx].set_ylim([0, 0.1])
    axes[ax_idx].set_yticks([0, 0.05, 0.1])
    axes[ax_idx].set_yticklabels(['$0.0$', '$0.05$', '$0.1$'])
    axes[ax_idx].set_xticks([1 + ax_idx * 4, 2 + ax_idx * 4, 3 + ax_idx * 4])
    axes[ax_idx].set_xticklabels(['true', 'IVA-G', 'PARAFAC2'], rotation=90)
    if filename is not None:
        axes[ax_idx].set_title(r'\Large{$\mathbf{a}^{[k]}_3$,} \normalsize{$k = 1, \dots, 4$}')
    else:
        axes[ax_idx].set_title(r'$\mathbf{a}^{[k]}_3, k = 1, \dots, 4$')

    ax_idx = 3
    axes[ax_idx].boxplot(p_values_true[:, 2, 4], positions=[1 + ax_idx * 4],
                         labels=['true'], boxprops=colors_true,
                         medianprops=colors_true, whiskerprops=colors_true,
                         capprops=colors_true, flierprops=dict(markeredgecolor=colors[0]))

    axes[ax_idx].boxplot(p_values_iva[:, 2, 4:].flatten(), positions=[2 + ax_idx * 4],
                         labels=['IVA-G'], boxprops=colors_iva,
                         medianprops=colors_iva, whiskerprops=colors_iva,
                         capprops=colors_iva, flierprops=dict(markeredgecolor=colors[1]))

    axes[ax_idx].boxplot(p_values_parafac2[:, 2], positions=[3 + ax_idx * 4],
                         labels=['PARAFAC2'], boxprops=colors_parafac2,
                         medianprops=colors_parafac2, whiskerprops=colors_parafac2,
                         capprops=colors_parafac2, flierprops=dict(markeredgecolor=colors[2]))

    axes[ax_idx].set_ylim([0, 1])
    axes[ax_idx].set_yticks([0, 0.5, 1])
    axes[ax_idx].set_yticklabels(['$0.0$', '$0.5$', '$1.0$'])
    axes[ax_idx].set_xticks([1 + ax_idx * 4, 2 + ax_idx * 4, 3 + ax_idx * 4])
    axes[ax_idx].set_xticklabels(['true', 'IVA-G', 'PARAFAC2'], rotation=90)
    if filename is not None:
        axes[ax_idx].set_title(r'\Large{$\mathbf{a}^{[k]}_3$,} \normalsize{$k=5, \dots, 12$}')
    else:
        axes[ax_idx].set_title(r'$\mathbf{a}^{[k]}_3, k=5, \dots, 12$')

    plt.tight_layout()
    if filename is not None:
        tikzplotlib.save(filename, encoding='utf8', axis_width='3.7cm')
    plt.show()
