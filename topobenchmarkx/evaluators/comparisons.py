import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


def signed_ranks_test(result1, result2):
    """Calculates the p-value for the Wilcoxon signed-rank test between the
    results of two models.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

    :param results: A 2xN numpy array with the results from the two models. N
        is the number of datasets over which the models have been tested on.
    :return: The p-value of the test
    """
    xs = result1 - result2
    return wilcoxon(xs[xs != 0])[1]


def friedman_test(results):
    """Calculates the p-value of the Friedman test between M models on N
    datasets.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html

    :param results: A MxN numpy array with the results of M models over N
        dataset
    :return: The p-value of the test
    """
    res = [r for r in results]
    return friedmanchisquare(*res)[1]


def compare_models(results, p_limit=0.05, verbose=False):
    """Compares different models. First it uses the Friedman test to check that
    the models are significantly different, then it uses pairwise comparisons
    to study the ranking of the models.

    :param results: A MxN numpy array with the results of M models over N
        dataset
    :param p_limit: The limit below which a hypothesis is considered false
    :param verbose: Whether to print the results of the tests or not
    :return average_rank: The average ranks of the models
    :return groups: List of lists with the groups of models that are
        statistically similar
    """
    M = results.shape[0]

    sorted_indices = np.argsort(-results, axis=0)
    ranks = np.empty_like(sorted_indices)
    for i in range(ranks.shape[1]):
        ranks[sorted_indices[:, i], i] = np.arange(len(results))+1
    average_ranks = np.mean(ranks, axis=1)
    
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
    print(signed_ranks_test(results[0, :], results[1, :]))
    results2 = np.array(
        [
            [0.7, 0.7, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.7, 0.9],
            [0.69, 0.71, 0.77, 0.9, 0.59, 0.811, 0.813, 0.78, 0.712, 0.899],
            [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.22, 0.32, 0.11, 0.4],
        ]
    )
    print(friedman_test(results2))
    results3 = np.array(
        [
            [0.9, 0.9, 0.8, 0.8, 0.7],
            [0.91, 0.89, 0.81, 0.79, 0.71],
            [0.89, 0.91, 0.79, 0.81, 0.69],
        ]
    )
    print(friedman_test(results3))

    print(compare_models(results2, verbose=True))
    print(compare_models(results3))
