"""Strict branch and hierarchy switches for paired, read-only policy ablations."""
import copy
import numpy as np

from assembly_context_policy import apply_policy
from assembly_context_features import number

MAIN_MODES=('N','C','NG','CG','NC','NCG')
TREE_LEVELS=('off','topology','on')


def global_contexts(rows):
    """No glTF hierarchy or names: scene scale only, with no implicit parent-size feature."""
    low=np.min([[number(r,'bbox_min_'+a)for a in 'xyz']for r in rows],axis=0)
    high=np.max([[number(r,'bbox_max_'+a)for a in 'xyz']for r in rows],axis=0)
    diagonal=max(float(np.linalg.norm(high-low)),1e-12)
    return {str(r['node_index']):dict(context_available=False,context=None,ancestor_candidates=[],
        topology_available=False,parent_index=None,depth=0,local_group_index=None,local_group_members=0,
        group_size_ratio=number(r,'bbox_diagonal')/diagonal,scene_size_ratio=number(r,'bbox_diagonal')/diagonal,
        local_mesh_repeat_count=1,scene_diagonal=diagonal)for r in rows}


def classify(row,opened,taxonomy,thresholds,descriptor,context,config,mode,tree='off',leaf=False):
    if mode not in (*MAIN_MODES,'G')or tree not in TREE_LEVELS:raise ValueError('Invalid factorial condition')
    source=dict(row)
    if 'N'not in mode:
        # The legacy context scorer may inspect these fields when checking material support.
        # Masking at the boundary prevents unavailable N evidence from entering C/G+H.
        source.update(predicted_class='',confidence=0.,second_confidence=0.)
    result=apply_policy(source,opened if 'C'in mode else {},taxonomy,thresholds,
        descriptor if 'G'in mode else {},copy.deepcopy(context),config,mode,
        use_tree=tree=='on',use_leaf=leaf,refinement='scoped')
    result.update(ablation_mode=mode,ablation_tree=tree,ablation_leaf_names=leaf,
        migration_geometry_guard=False,decision_inputs_scope='enabled branches only; H0 removes all parent context')
    return result
