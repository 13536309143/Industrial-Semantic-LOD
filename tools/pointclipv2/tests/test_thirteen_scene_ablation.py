import csv
import gzip
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_thirteen_scene_ablation import digest, valid_stamp, save
from analyze_thirteen_scene_ablation import pair, table
from thirteen_scene_coverage import align_mesh_population


class ThirteenSceneTests(unittest.TestCase):
    def test_empty_mesh_is_audited_without_fabricating_a_prediction(self):
        data=dict(nodes=[dict(mesh=0),dict(mesh=1)],meshes=[dict(primitives=[dict(attributes={'POSITION':0})]),dict(primitives=[])],accessors=[dict(count=3)])
        a=[dict(node_index='0',mesh_index='0')]
        b=[dict(node_index='0',mesh_index='0',open_status='accepted'),dict(node_index='1',mesh_index='1',open_status='abstain')]
        pn,opened,excluded=align_mesh_population(data,a,b)
        self.assertIs(pn,a);self.assertEqual(opened,b[:1]);self.assertEqual(excluded[0]['reason'],'no_POSITION_vertices')

    def test_missing_nonempty_supervised_mesh_is_not_silently_excluded(self):
        data=dict(nodes=[dict(mesh=0)],meshes=[dict(primitives=[dict(attributes={'POSITION':0})])],accessors=[dict(count=3)])
        with self.assertRaisesRegex(AssertionError,'supervised'):
            align_mesh_population(data,[],[dict(node_index='0',mesh_index='0',open_status='abstain')])

    def test_status_publication_retries_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'status.json'
            target.write_text('{"status":"old"}',encoding='utf-8')
            original=Path.replace;attempts=[]
            def replace(source,destination):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('temporary reader lock')
                return original(source,destination)
            with patch.object(Path,'replace',replace),patch('run_thirteen_scene_ablation.time.sleep'):
                save(target,{'status':'complete'})
            self.assertEqual(len(attempts),3)
            self.assertEqual(json.loads(target.read_text())['status'],'complete')

    def test_limited_smoke_result_is_not_reused_as_full_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            stamp=Path(folder)/'proof.json'
            stamp.write_text(json.dumps(dict(inputs={'old.glb':'scene'},outputs={},parameters={'limit':2})),encoding='utf-8')
            self.assertFalse(valid_stamp(stamp,'scene'))

    def test_renamed_scene_reuse_requires_all_other_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);code=root/'code.py';output=root/'output.csv';stamp=root/'proof.json'
            code.write_text('frozen',encoding='utf-8');output.write_text('result',encoding='utf-8')
            stamp.write_text(json.dumps(dict(inputs={str(root/'old-name.glb'):'scene-bytes',str(code):digest(code)},
                outputs={str(output):digest(output)})),encoding='utf-8')
            self.assertTrue(valid_stamp(stamp,'scene-bytes'))
            self.assertFalse(valid_stamp(stamp,'different-scene'))
            code.write_text('changed source',encoding='utf-8')
            self.assertFalse(valid_stamp(stamp,'scene-bytes'))

    def test_modified_output_is_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'output.csv';stamp=root/'proof.json'
            output.write_text('original',encoding='utf-8')
            stamp.write_text(json.dumps(dict(inputs={str(root/'renamed.glb'):'identical'},
                outputs={str(output):digest(output)})),encoding='utf-8')
            output.write_text('corrupted result',encoding='utf-8')
            self.assertFalse(valid_stamp(stamp,'identical'))

    def test_gzip_csv_preserves_unicode_and_optional_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'nodes.csv.gz'
            table(target,[{'node_name':'\u5f39\u7c27','node_index':'10'},
                          {'node_name':'part,with,commas','node_index':'11','detail':'extra'}])
            with gzip.open(target,'rt',encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['node_name'],'\u5f39\u7c27')
            self.assertEqual(rows[1]['detail'],'extra')

    def test_local_constraint_change_is_reported_without_priority_change(self):
        a=dict(order=1,node_index=7,mesh_index=2,node_name='sample',face_count=100,
            lod_priority='P4_normal',function_role='unknown',resolved_object_class='',
            target_ratio_far=.2,feature_lock_budget=.3)
        b=dict(a,feature_lock_budget=.6)
        result,matrix,changes=pair('M1','NCG',[a],[b],'H0','H1','main')
        self.assertEqual(result['priority_changed'],0)
        self.assertEqual(result['action_changed'],1)
        self.assertEqual(sum(r['count'] for r in matrix),1)
        self.assertEqual(len(changes),1)

    def test_unpaired_node_keys_are_rejected(self):
        with self.assertRaises(AssertionError):
            pair('M1','N',[dict(order=1,node_index=7,mesh_index=2)],
                 [dict(order=1,node_index=8,mesh_index=2)],'H0','H1','main')


if __name__=='__main__':unittest.main()
