import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def signed_ranks_test(results_1, results_2):
    r"""Calculates the p-value for the Wilcoxon signed-rank test between the
    results of two models.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

    Args:
        results_1 (numpy.array): A numpy array with the results from the first model. N
        is the number of datasets over which the models have been tested on.
        results_2 (numpy.array): A numpy array with the results from the second model. Needs to have the same shape as results_1.
    Returns:
        float: The p-value of the test.
    """
    xs = results_1 - results_2
    xs = xs[~np.isnan(xs)]
    return wilcoxon(xs[xs != 0])[1]


def friedman_test(results):
    r"""Calculates the p-value of the Friedman test between M models on N
    datasets.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html

    Args:
        results (numpy.array): A MxN numpy array with the results of M models.
    Returns:
        float: The p-value of the test.
    """
    res = [r for r in results]
    return friedmanchisquare(*res)[1]


def skillings_mack_test(results, n_simulations=10000):
    r"""Calculates the p-value of the Skillings-Mack test between M models
    (treatments) on N datasets (blocks). This test is a general Friedman-type
    statistic that can be used in almost any block design with an arbitrary
    missing-data structure.

    https://www.stat.cmu.edu/technometrics/80-89/VOL-23-02/v2302171.pdf

    Args:
        results (numpy.array): A MxN numpy array with the results of M models.
        n_simulations (int, optional): The number of simulations to perform to estimate the p-value. (default: 10000)
    Returns:
        float: The p-value of the test.
    """

    def get_SM_value(results):
        r"""Calculates the Skillings-Mack statistic for a given set of results.

        Args:
            results (numpy.array): A MxN numpy array with the results of M models.
        Returns:
            float: The Skillings-Mack statistic.
        """
        M, N = results.shape
        cols_to_remove = []
        for i in range(N):
            if np.sum(~np.isnan(results[:, i])) <= 1:
                cols_to_remove.append(i)
        results = np.delete(results, cols_to_remove, axis=1)
        k = np.sum(~np.isnan(results), axis=0)

        ranks = rankdata(-results, axis=0, method="average", nan_policy="omit")

        A = np.zeros(M)
        for j in range(M):
            for i in range(N):
                if ~np.isnan(ranks[j, i]):
                    A[j] += np.power(12 / (k[i] + 1), 0.5) * (
                        ranks[j, i] - (k[i] + 1) / 2
                    )

        covariance_matrix = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i != j:
                    covariance_matrix[i, j] = -np.sum(
                        ~np.isnan(results[i]) & ~np.isnan(results[j])
                    )
        for i in range(M):
            covariance_matrix[i, i] = -np.sum(covariance_matrix[i, :])
        rank = np.linalg.matrix_rank(covariance_matrix)
        if rank == M - 1:
            covariance_matrix = np.pad(
                covariance_matrix[:-1, :-1], ((0, 1), (0, 1)), mode="constant"
            )
        else:
            raise ValueError(
                f"Problem with the covariance matrix having rank {rank} instead of {M-1} (M-1)."
            )
        inv_covariance_matrix = np.linalg.pinv(covariance_matrix)
        SM = np.dot(A.T, np.dot(inv_covariance_matrix, A))
        return SM

    def get_simulated_SM_values(res, mask_nans, n_simulations):
        r"""Calculates the Skillings-Mack statistic for the null hypothesis. The
        null hypothesis is that the models are equivalent and the differences
        between them are due to random noise. For this reason we use the normal
        distribution to obtain the simulated results.

        Args:
            res (numpy.array): A MxN numpy array with the results of M models.
            n_nans (numpy.array): A numpy array with the number of NaNs per row.
            n_simulations (int): The number of simulations to perform.
        """
        res_shape = res.shape
        SMs = np.zeros(n_simulations)
        for i in range(n_simulations):
            res = np.random.normal(loc=0, scale=1, size=res_shape)
            res[mask_nans] = np.nan
            SMs[i] = get_SM_value(res)
        return SMs

    SM = get_SM_value(results)
    mask_nans = np.isnan(results)
    null_SM = get_simulated_SM_values(results, mask_nans, n_simulations)
    p = np.sum(null_SM > SM) / n_simulations

    return p


def compare_models(results, p_limit=0.05, verbose=False):
    """Compares different models. First it uses the Friedman test or the
    Skillings-Mack test to check that the models are significantly different,
    then it uses pairwise comparisons to study the ranking of the models.

    Args:
        results (numpy.array): A MxN numpy array with the results of M models
        over N dataset.
        p_limit (float, optional): The limit below which a hypothesis is considered false. (default: 0.05)
        verbose (bool, optional): Whether to print the results of the tests or not. (default: False)
    Returns:
        numpy.array: The average ranks of the models
        list: A list of lists with the indices of the models that are in the same group. The first group is the best one.
    """
    M = results.shape[0]

    ranks = rankdata(-results, axis=0, nan_policy="omit")

    k = np.sum(~np.isnan(results), axis=0)
    for i in range(M):
        ranks[i, np.isnan(results[i])] = (k[i] + 1) / 2
    average_ranks = np.mean(ranks, axis=1)

    if np.any(np.isnan(results)):
        p_value = skillings_mack_test(results, n_simulations=10000)
        test_name = "Skillings-Mack"
    else:
        p_value = friedman_test(results)
        test_name = "Friedman"
    if p_value > p_limit:
        if verbose:
            print(
                f"The {test_name} test confirmed the null-hypothesis with p-value of {p_value}"
            )
        return average_ranks, [[i for i in range(M)]]

    groups = []
    for i in range(M):
        idx = 1
        rankings = rankdata(average_ranks) - 1
        model_idx = np.where(rankings == i)[0][0]
        group = [model_idx]
        while i + idx < M:
            next_model_idx = np.where(rankings == i + idx)[0][0]
            p = signed_ranks_test(
                results[model_idx, :], results[next_model_idx, :]
            )
            if verbose:
                print(
                    f"P-value for Wilcoxon test between models {model_idx} and {next_model_idx}: {p}"
                )
            if p < p_limit:
                break
            group.append(next_model_idx)
            idx += 1
        groups.append(group)
    return average_ranks, groups


if __name__ == "__main__":
    results = np.array(
        [
            [0.7, 0.7, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.7, 0.9],
            [0.6, 0.65, 0.7, 0.9, 0.5, 0.552, 0.843, 0.78, 0.665, 0.876],
        ]
    )
    print("Signed ranks test:")
    print(signed_ranks_test(results[0, :], results[1, :]))
    results2 = np.array(
        [
            [0.7, 0.7, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.7, 0.9],
            [0.69, 0.71, 0.77, 0.9, 0.59, 0.811, 0.813, 0.78, 0.712, 0.899],
            [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.22, 0.32, 0.11, 0.4],
        ]
    )
    print("Friedman test with very different results:")
    print(friedman_test(results2))
    results3 = np.array(
        [
            [0.9, 0.9, 0.8, 0.8, 0.7],
            [0.91, 0.89, 0.81, 0.79, 0.71],
            [0.89, 0.91, 0.79, 0.81, 0.69],
        ]
    )
    print("Friedman test with similar results:")
    print(friedman_test(results3))

    print("-" * 50)
    print("Compare models with different results:")
    print(compare_models(results2, verbose=True))
    print("-" * 50)
    print("Compare models with similar results:")
    print(compare_models(results3, verbose=True))

    results4 = np.array(
        [
            [3.2, 3.1, 4.3, 3.5, 3.6, 4.5, np.nan, 4.3, 3.5],
            [4.1, 3.9, 3.5, 3.6, 4.2, 4.7, 4.2, 4.6, np.nan],
            [3.8, 3.4, 4.6, 3.9, 3.7, 3.7, 3.4, 4.4, 3.7],
            [4.2, 4.0, 4.8, 4.0, 3.9, np.nan, np.nan, 4.9, 3.9],
        ]
    )
    skillings_mack_test(-results4)
