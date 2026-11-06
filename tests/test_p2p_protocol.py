import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/geotransformer.p2p_liver'
sys.path.insert(0, str(EXPERIMENT))
from config import make_cfg
from dataset import LiverTask3TrainDataset, DeterministicValidationDataset
from model import create_model


class ProtocolTest(unittest.TestCase):
    def test_rovenet_always_contains_both_refiners(self):
        model = create_model(make_cfg())
        names = list(model.state_dict())
        self.assertEqual(type(model).__name__, 'ROVENet')
        self.assertTrue(any(n.startswith('topology_overlap_refiner.') for n in names))
        self.assertTrue(any(n.startswith('fine_local_refiner.') for n in names))
        self.assertFalse(hasattr(model, 'rtor_enabled'))
        self.assertFalse(hasattr(model, 'a3_enabled'))

    def test_visibility_edges_are_counted_once(self):
        spec = importlib.util.spec_from_file_location('p2p_eval', EXPERIMENT / 'test.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = np.ones(9)
        rows = module._visibility_summary(values, np.array([i/10 for i in range(2, 11)]))
        self.assertEqual([r['count'] for r in rows], [1]*7 + [2])

    def test_validation_restores_random_state(self):
        import random
        class RandomDataset:
            def __getitem__(self, index):
                return random.random(), np.random.rand()
        data = DeterministicValidationDataset(RandomDataset(), 2, 123)
        random.seed(7)
        np.random.seed(7)
        before = random.getstate(), np.random.get_state()
        first = data[0]
        self.assertEqual(first, data[0])
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])


if __name__ == '__main__':
    unittest.main()
