import unittest
import numpy as np
import engine as E

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = E.Config(n=1000, seed=7)

    def test_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError): E.Config(poor_share=1.5)
        with self.assertRaises(ValueError): E.Config(w_fingerprint=0, w_iris=0, w_face=0)

    def test_weights_normalize(self):
        self.assertAlmostEqual(sum(self.cfg.weights().values()), 1.0)

    def test_scores_are_bounded(self):
        scores, quality = E.genuine(np.random.default_rng(1), 1000, .2)
        for trait in E.TRAITS:
            self.assertTrue(np.all((scores[trait] >= 0) & (scores[trait] <= 1)))
            self.assertTrue(np.all((quality[trait] >= 0) & (quality[trait] <= 1)))

    def test_liveness_failure_cannot_enter_fallback(self):
        spec = E.Spec("test", "qw", liveness=True, fallback=True)
        method = E.Method(self.cfg, spec, np.random.default_rng(2))
        scores, quality = E.genuine(np.random.default_rng(3), 10, .2)
        direct, escalated, _, _ = method.decide(scores, quality, np.ones(10, dtype=bool))
        self.assertFalse(direct.any())
        self.assertFalse(escalated.any())

    def test_single_attempt_is_repeatable(self):
        spec = E.BASELINE_SPECS[-1]
        a = E.single_attempt(self.cfg, spec, "genuine", seed=123)
        b = E.single_attempt(self.cfg, spec, "genuine", seed=123)
        self.assertEqual(a, b)

    def test_spoof_quality_override_does_not_turn_unspoofed_trait_genuine(self):
        spec = E.Spec("test", "qw")
        result = E.single_attempt(self.cfg, spec, "spoof", attack="dummy", scenario="partial",
                                  qualities={"face": .9}, seed=44)
        # In a dummy-finger attack, face remains an impostor trait.
        self.assertLess(result["scores"]["face"], .8)

    def test_evaluation_metrics_are_finite_and_bounded(self):
        result = E.evaluate(self.cfg, E.BASELINE_SPECS[-1], np.random.default_rng(9))
        for key, value in result.items():
            if key != "method":
                self.assertTrue(np.isfinite(value), key)
                self.assertGreaterEqual(value, 0, key)
                self.assertLessEqual(value, 100, key)

    def test_method_runs_repeatably(self):
        a = E.run_methods(self.cfg, E.BASELINE_SPECS)
        b = E.run_methods(self.cfg, E.BASELINE_SPECS)
        self.assertTrue(a.equals(b))

if __name__ == "__main__":
    unittest.main()
