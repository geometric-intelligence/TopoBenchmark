import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


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


def compare_models(results, p_limit=0.05, verbose=False):
    """Compares different models. First it uses the Friedman test to check that
    the models are significantly different, then it uses pairwise comparisons
    to study the ranking of the models.

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

    average_ranks = np.mean(np.argsort(-results2, axis=0) + 1, axis=1)

    friedman_result = friedman_test(results)
    if friedman_result > p_limit:
        if verbose:
            print(
                f"The Friedman test confirmed the null-hypothesis with p-value of {friedman_result}"
            )
        return average_ranks, [[i for i in range(M)]]

    groups = []
    for i in range(M):
        idx = 1
        model_idx = np.where(np.argsort(average_ranks) == i)[0][0]
        group = [model_idx]
        while i + idx < M:
            next_model_idx = np.where(np.argsort(average_ranks) == i + idx)[0][
                0
            ]
            p = signed_ranks_test(
                results[model_idx, :], results[model_idx + idx, :]
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

    print("-"*50)
    print("Compare models with different results:")
    print(compare_models(results2, verbose=True))
    print("-"*50)
    print("Compare models with similar results:")
    print(compare_models(results3, verbose=True))
