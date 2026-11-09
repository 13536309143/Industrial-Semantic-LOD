"""Resolve glTF parent transforms without changing source node identifiers."""
import numpy as np


def attach_world_matrices(gltf, local_matrix):
    """Attach runtime matrices; reject cycles/multiple parents rather than guess."""
    nodes = gltf.get("nodes", [])
    parents = [None] * len(nodes)
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            if not isinstance(child, int) or not 0 <= child < len(nodes):
                raise ValueError(f"Invalid child node {child}")
            if parents[child] is not None:
                raise ValueError(f"Node {child} has multiple parent references")
            parents[child] = parent
    matrices = [None] * len(nodes)
    for index in range(len(nodes)):
        chain, visited, cursor = [], set(), index
        while cursor is not None and matrices[cursor] is None:
            if cursor in visited:
                raise ValueError(f"Cycle in glTF hierarchy at node {cursor}")
            visited.add(cursor)
            chain.append(cursor)
            cursor = parents[cursor]
        world = np.eye(4) if cursor is None else matrices[cursor]
        for item in reversed(chain):
            world = world @ local_matrix(nodes[item])
            if not np.isfinite(world).all():
                raise ValueError(f"Nonfinite node transform: {item}")
            matrices[item] = world
    for index, node in enumerate(nodes):
        node["_world_matrix"] = matrices[index]
        node["_parent_index"] = parents[index]
    return parents
