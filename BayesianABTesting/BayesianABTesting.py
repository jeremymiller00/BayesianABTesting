from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


class BayesianABTesting:

    def __init__(self, likelihood_function: str, data: dict):
        if likelihood_function not in ["binomial"]:
            raise NotImplementedError("This functionality not yet implemented")
        self.likelihood_function = likelihood_function
        self.data = data

    def execute_test(self, metric: str = "Metric", verbose=1, *diffs):
        if self.likelihood_function == "binomial":
            return self._test_binom(metric, verbose, *diffs)
        else:
            raise NotImplementedError("This functionality not yet implemented")

    def _test_binom(self, metric: str, verbose: int, plot=True, labels=None, *diffs):
        """
        # diffs is used to pass values for which to calculate the probability of a difference at least that large
        # returns most probable difference, and p of that difference, as well as p difference is 0

        :param metric:
        :param verbose:
        :param diffs:
        :return:
        """
        a_failures = self.data.get("a_trials") - self.data.get("a_successes")
        b_failures = self.data.get("b_trials") - self.data.get("b_successes")

        # here are our posterior distributions
        beta_a = stats.beta(1 + self.data.get("a_successes"), 1 + a_failures)
        beta_b = stats.beta(1 + self.data.get("b_successes"), 1 + b_failures)
        sample_a = beta_a.rvs(size=100000)
        sample_b = beta_b.rvs(size=100000)
        result_1 = (sample_a < sample_b).mean()
        result_2 = (sample_a > sample_b).mean()
        diff = beta_b.mean() - beta_a.mean()
        if diff > 0.0:
            winner = "B"
        else:
            winner = "A"
        prob = ((sample_a + diff) < sample_b).mean()
        if verbose > 0:
            print(
                f"Probability that A {metric} is less than B {metric} is approximately {result_1}."
            )
            print(
                f"Probability that A {metric} is greater than B {metric} is approximately {result_2}."
            )
            print(
                f"Most likely difference: {np.abs(round(diff, 4))} with {winner} being greater"
            )
            print(
                f"Probability of most likely difference: {round(prob, 4)}"
            )
        if plot:
            fig = self._plot_posteriors(self.data.get("a_successes"), a_failures,
                                        self.data.get("b_successes"), b_failures,
                                        labels)
            return winner, diff, prob, fig

        return winner, diff, prob

    def _plot_posteriors(self, a_successes, a_failures, b_successes, b_failures, labels=["A", "B"]):
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.linspace(0, 1, 10000)
        for (a, b, s, label) in [(1 + a_successes, 1 + a_failures, "r", labels[0]),
                                 (1 + b_successes, 1 + b_failures, "b", labels[1])]:
            ax.plot(x,
                    stats.beta(a, b).pdf(x),
                    s,
                    label=label)
        ax.legend(loc="upper right")
        ax.set_xlabel("p")
        ax.set_ylabel("pdf")
        ax.set_title(f"Posterior Probabilities Selection Rate for API {labels[0]} vs API {labels[1]}",
                     fontsize=20)
        return fig







