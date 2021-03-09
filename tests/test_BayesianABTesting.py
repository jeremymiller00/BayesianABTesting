from unittest import TestCase
from BayesianABTesting.BayesianABTesting import BayesianABTesting


class TestBayesianABTesting(TestCase):
    def setUp(self) -> None:
        self.data = {
            "a_trials": 100,
            "a_successes": 10,
            "b_trials": 1000,
            "b_successes": 120
        }
        self.likelihood_function = "binomial"
        self.tester = BayesianABTesting(self.likelihood_function, self.data)

    def test__test_binom(self):
        winner, diff, prob = self.tester._test_binom("metric", 1)
        self.assertIsInstance(winner, str, "Invalid type, should be str")
        self.assertIsInstance(diff, float, "Invalid numeric type")
        self.assertIsInstance(prob, float, "Invalid numeric type")
        self.assertIn(winner, ["A", "B"], "Invalid value for winner")
        self.assertAlmostEqual(diff, 0.0129, delta=0.0001)
        self.assertAlmostEqual(prob, 0.53, delta=0.01)
