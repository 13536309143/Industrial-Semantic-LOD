import sys
import unittest
import json
import mmap
import struct
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pointnext_lod"))
from gltf_hierarchy import attach_world_matrices


def local(node):
    if "matrix" in node:
        return np.array(node["matrix"]).reshape(4, 4).T
    result = np.eye(4)
    result[:3, :3] = np.diag(node.get("scale", [1, 1, 1]))
    result[:3, 3] = node.get("translation", [0, 0, 0])
    return result


class HierarchyTests(unittest.TestCase):
    def test_scene_triangle_audit_counts_reused_mesh_instances(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from analyze_scene_semantics import glb_structure
        data = {'nodes': [{'mesh': 0}, {'mesh': 0}, {'mesh': 1}],
                'meshes': [{'primitives': [{'indices': 0}]}, {'primitives': [{'indices': 1}]}],
                'accessors': [{'count': 9}, {'count': 6}]}
        text = json.dumps(data).encode()
        text += b' ' * (-len(text) % 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'instances.glb'
            path.write_bytes(struct.pack('<3I', 0x46546c67, 2, 20 + len(text)) +
                             struct.pack('<2I', len(text), 0x4e4f534a) + text)
            stats = glb_structure(path)
            self.assertEqual(stats['unique_mesh_triangles'], 5)
            self.assertEqual(stats['source_triangles'], 8)

    def test_both_inference_readers_decode_same_world_geometry(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import run_pointclipv2_zeroshot_glb as clip_reader
        import analyze_large_glb_parts_pointnext as next_reader
        data = {'nodes': [{'children': [1], 'translation': [30, 0, 0], 'scale': [2, 2, 2]},
                          {'mesh': 0, 'translation': [1, 0, 0]}],
                'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],
                'bufferViews': [{'byteOffset': 0, 'byteLength': 36}, {'byteOffset': 36, 'byteLength': 6}],
                'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3'},
                              {'bufferView': 1, 'componentType': 5123, 'count': 3, 'type': 'SCALAR'}]}
        text = json.dumps(data).encode()
        text += b' ' * (-len(text) % 4)
        binary = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype='<f4').tobytes() + struct.pack('<3H', 0, 1, 2) + b'\0\0'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested.glb'
            path.write_bytes(struct.pack('<3I', 0x46546c67, 2, 28 + len(text) + len(binary)) +
                             struct.pack('<2I', len(text), 0x4e4f534a) + text +
                             struct.pack('<2I', len(binary), 0x004e4942) + binary)
            for reader in [clip_reader, next_reader]:
                gltf, offset, _ = reader.read_glb_chunks(path)
                with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    vertices, faces = reader.mesh_for_node(gltf, mm, offset, gltf['nodes'][1])
                    np.testing.assert_allclose(vertices, [[32, 0, 0], [34, 0, 0], [32, 2, 0]])
                    np.testing.assert_array_equal(faces, [[0, 1, 2]])

    def test_nested_parent_scale_rotation_translation_and_mesh_instances(self):
        root = np.array([[0, -2, 0, 10], [2, 0, 0, 20], [0, 0, 2, 0], [0, 0, 0, 1]])
        data = {"nodes": [{"children": [1, 2], "matrix": root.T.flatten().tolist()},
                          {"mesh": 0, "translation": [1, 0, 0]},
                          {"mesh": 0, "translation": [0, 2, 0]}]}
        attach_world_matrices(data, local)
        np.testing.assert_allclose(data["nodes"][1]["_world_matrix"][:3, 3], [10, 22, 0])
        np.testing.assert_allclose(data["nodes"][2]["_world_matrix"][:3, 3], [6, 20, 0])
        self.assertEqual(data["nodes"][1]["_parent_index"], 0)

    def test_flat_nodes_keep_original_local_transform(self):
        data = {"nodes": [{"translation": [4, 5, 6]}]}
        attach_world_matrices(data, local)
        np.testing.assert_allclose(data["nodes"][0]["_world_matrix"], local(data["nodes"][0]))

    def test_invalid_trees_fail_explicitly(self):
        for nodes in [[{"children": [1]}, {"children": [0]}],
                      [{"children": [2]}, {"children": [2]}, {}]]:
            with self.assertRaises(ValueError):
                attach_world_matrices({"nodes": nodes}, local)


if __name__ == "__main__":
    unittest.main()
