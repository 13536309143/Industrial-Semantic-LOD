import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reuse_scene_inference as snapshot
from artifact_cache import file_hash, files_identity, record, stage_identity


class SnapshotReuseTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'old'
        source.mkdir()
        model = root / 'models/pointnext/industrial-part-14class-best.pth'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'test-checkpoint')
        glb, code = root / 'part.glb', root / 'source.py'
        glb.write_bytes(b'test-scene')
        code.write_text('version=1')
        state = {'glb_sha256': file_hash(glb), 'checkpoint_sha256': file_hash(model),
                 'stages': [{'name': 'test', 'wall_seconds': 12}], 'status': 'inference_complete_analysis_pending'}
        (source / 'inference_status.json').write_text(json.dumps(state), encoding='utf-8')
        for suffix in ['pointnext_provenance', 'full.provenance', 'candidates.provenance']:
            output = source / (suffix + '.csv')
            output.write_bytes(b'a,b\n1,2\n')
            inputs = [glb, code, model]
            if suffix == 'candidates.provenance':
                inputs.append(source / 'pointnext_provenance.csv')
            record(source / f'part_{suffix}.json', stage_identity(inputs, {'points': 8192}), [output])
        return source, glb, code

    def test_copy_keeps_source_and_relocates_verified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, glb, code = self.fixture(root)
            original = file_hash(source / 'inference_status.json')
            target = root / 'new'
            with patch.object(snapshot, 'ROOT', root):
                snapshot.reuse(source, target, glb)
            self.assertEqual(file_hash(source / 'inference_status.json'), original)
            payload = json.loads((target / 'part_candidates.provenance.json').read_text(encoding='utf-8'))
            self.assertIn(str((target / 'pointnext_provenance.csv').resolve()), payload['inputs'])
            self.assertEqual(payload['outputs'], files_identity(payload['outputs']))
            self.assertTrue(json.loads((target / 'inference_status.json').read_text(encoding='utf-8'))['stages'][0]['reused_snapshot'])

    def test_changed_code_refuses_reuse_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, glb, code = self.fixture(root)
            code.write_text('version=2')
            with patch.object(snapshot, 'ROOT', root), self.assertRaises(ValueError):
                snapshot.reuse(source, root / 'new', glb)
            self.assertFalse((root / 'new').exists())

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, glb, code = self.fixture(root)
            with patch.object(snapshot, 'ROOT', root), self.assertRaises(FileExistsError):
                snapshot.reuse(source, source, glb)


if __name__ == '__main__':
    unittest.main()
