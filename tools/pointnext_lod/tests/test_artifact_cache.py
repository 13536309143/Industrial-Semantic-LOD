"""Regression checks for checkpoint swaps, interrupted outputs and legacy CSVs."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_cache import matches, record, stage_identity


class ArtifactCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.checkpoint = self.root / "model.pth"
        self.checkpoint.write_bytes(b"old-model")
        self.output = self.root / "scene.csv"
        self.output.write_text("order,predicted\n1,nuts\n")
        self.stamp = self.root / "stage.json"

    def identity(self, points=8192):
        return stage_identity([self.checkpoint], {"points": points})

    def test_checkpoint_and_sampling_change_invalidate_completed_outputs(self):
        identity = self.identity()
        record(self.stamp, identity, [self.output])
        self.assertTrue(matches(self.stamp, identity, [self.output]))
        self.assertFalse(matches(self.stamp, self.identity(4096), [self.output]))
        self.checkpoint.write_bytes(b"new-model")
        self.assertFalse(matches(self.stamp, self.identity(), [self.output]))

    def test_unstamped_legacy_and_truncated_outputs_are_not_reused(self):
        identity = self.identity()
        self.assertFalse(matches(self.stamp, identity, [self.output]))
        record(self.stamp, identity, [self.output])
        self.output.write_text("order,predicted\n")
        self.assertFalse(matches(self.stamp, identity, [self.output]))
        self.output.unlink()
        self.assertFalse(matches(self.stamp, identity, [self.output]))

    def test_candidate_csv_and_prompt_changes_invalidate_open_stage(self):
        prompt = self.root / "prompts.json"
        prompt.write_text('{"labels": ["nuts"]}')
        inputs = [self.checkpoint, prompt]
        identity = stage_identity(inputs, {})
        record(self.stamp, identity, [self.output])
        prompt.write_text('{"labels": ["springs"]}')
        self.assertFalse(matches(self.stamp, stage_identity(inputs, {}), [self.output]))


if __name__ == "__main__":
    unittest.main()
