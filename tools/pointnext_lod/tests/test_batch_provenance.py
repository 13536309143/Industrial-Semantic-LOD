"""Exercise batch resume without CUDA, including a checkpoint replacement."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch_analyze_glb_lod as batch


class BatchProvenanceTests(unittest.TestCase):
    def test_resume_rebuilds_legacy_outputs_and_invalidates_lod_after_swap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            resources = root / "scenes"
            resources.mkdir()
            (resources / "part.glb").write_bytes(b"fixture scene")
            output = root / "out"
            output.mkdir()
            (output / "part_pointnext_analysis.csv").write_text("legacy output")
            backbone = root / "PointNeXt/openpoints"
            backbone.mkdir(parents=True)
            (backbone / "model.py").write_text("# fixture encoder")
            for name in ["model.pth", "cfg.yaml", "classes.txt"]:
                (root / name).write_text("first version")
            arguments = ["batch", "--resources-dir", str(resources), "--output-dir", str(output),
                         "--pointnext-root", str(backbone.parent), "--cfg", str(root / "cfg.yaml"),
                         "--ckpt", str(root / "model.pth"), "--classes-file", str(root / "classes.txt"),
                         "--skip-existing"]
            calls = []

            def fake_inference(command, cwd):
                calls.append(Path(command[1]).name)
                content = (root / "model.pth").read_text()
                for flag in ["--output-csv", "--output-json", "--summary-csv", "--review-csv", "--summary-json"]:
                    if flag in command:
                        Path(command[command.index(flag) + 1]).write_text(content)

            with patch.object(sys, "argv", arguments), patch.object(batch, "run", fake_inference), contextlib.redirect_stdout(io.StringIO()):
                batch.main()
                self.assertEqual(len(calls), 2)  # legacy CSV must not be reused
                batch.main()
                self.assertEqual(len(calls), 2)  # completed, identical stages reuse
                (root / "model.pth").write_text("replacement weights")
                batch.main()
                self.assertEqual(len(calls), 4)  # both classifier and LOD rebuild
                (output / "part_pointnext_analysis.csv").write_text("truncated")
                batch.main()
                self.assertEqual(len(calls), 5)  # repairs classifier; identical restored LOD is valid


if __name__ == "__main__":
    unittest.main()
