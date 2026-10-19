"""A compatible classifier must never silently keep random missing parameters."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_loading import load_classifier_checkpoint


class CheckpointLoadingTests(unittest.TestCase):
    def test_complete_prefixed_checkpoint_loads_all_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            original = torch.nn.Linear(3, 14)
            checkpoint = Path(folder) / "model.pth"
            torch.save({"model": {"module." + k: v for k, v in original.state_dict().items()}, "epoch": 98}, checkpoint)
            restored = torch.nn.Linear(3, 14)
            load_classifier_checkpoint(restored, checkpoint)
            for key, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[key]))

    def test_missing_or_wrong_shape_classifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pth"
            torch.save({"model": {"weight": torch.zeros(14, 3)}}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_classifier_checkpoint(torch.nn.Linear(3, 14), checkpoint)
            torch.save({"model": torch.nn.Linear(3, 40).state_dict()}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_classifier_checkpoint(torch.nn.Linear(3, 14), checkpoint)


if __name__ == "__main__":
    unittest.main()
