"""Cross-check depth and surface shading without forcing disagreeing identities."""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

from open_semantic_evidence import infer_evidence,serialize_evidence
from run_open_semantic_v2 import write_rows
from run_pointclipv2_zeroshot_glb import read_glb_chunks
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/pointnext_lod'))
from artifact_cache import stage_identity,record


def combine(depth,surface):
    valid=lambda evidence:evidence.get('status')in {'accepted','class_only','family_only'}
    d,s=valid(depth),valid(surface)
    selected=copy.deepcopy(surface if surface.get('reliability',0)>depth.get('reliability',0)else depth)
    selected.update(depth_status=depth['status'],surface_status=surface['status'],
                    depth_family=depth.get('family',''),surface_family=surface.get('family',''))
    if d and s and depth.get('family')==surface.get('family'):
        selected['render_agreement']='family_consensus'
        selected['reliability']=min(.95,(depth['reliability']+surface['reliability'])/2+.05)
        selected['reason']='independent_depth_surface_family_consensus'
        fine=depth['status']=='accepted' and surface['status']=='accepted' and depth.get('object_id')==surface.get('object_id')
        class_consensus=bool(depth.get('class_group')) and depth.get('class_group')==surface.get('class_group')
        group_support=depth if class_consensus else None
        # A coarse, noncontradictory second stream need not erase a strong object-group decision.
        if not class_consensus and bool(depth.get('class_group')) != bool(surface.get('class_group')):
            strong=depth if depth.get('class_group') else surface
            if strong.get('class_group_margin',0)>=.01 and strong.get('class_group_agreement',0)>=.5:
                group_support=strong
        selected['status']='accepted' if fine else 'class_only' if group_support else 'family_only'
        selected['object_id']=depth.get('object_id','') if fine else ''
        selected['class_group']=group_support.get('class_group','') if group_support else ''
        selected['class_group_known_class']=group_support.get('class_group_known_class','') if group_support else ''
        selected['roles']={role:(depth.get('roles',{}).get(role,0)+surface.get('roles',{}).get(role,0))/2
                           for role in set(depth.get('roles',{}))|set(surface.get('roles',{}))}
    elif d and s:
        selected.update(status='abstain',object_id='',roles={},render_agreement='conflict',reason='depth_surface_family_conflict')
    elif d or s:
        supported=copy.deepcopy(depth if d else surface)
        if supported['reliability']>=.65 and supported.get('family_margin',0)>=.01:
            selected.update(supported)
            selected.update(status='family_only',object_id='',render_agreement='single_stream_supported',
                            reliability=supported['reliability']*.75,reason='single_render_family_only')
        else:
            selected.update(status='abstain',object_id='',roles={},render_agreement='unsupported',reason='single_render_insufficient_support')
    else:
        selected.update(status='abstain',object_id='',roles={},render_agreement='both_rejected',reason='both_renderings_abstained')
    if selected['status'] not in {'accepted','class_only'}:
        selected['class_group']='';selected['class_group_known_class']=''
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glb',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    directory=Path(__file__).parent
    taxonomy=json.loads((directory/'industrial_open_vocab_v2.json').read_text(encoding='utf-8'))
    config=json.loads((directory/'open_semantic_v2_config.json').read_text(encoding='utf-8'))
    gltf,_,_=read_glb_chunks(args.glb)
    nodes=[(i,node)for i,node in enumerate(gltf['nodes'])if 'mesh'in node]
    with np.load(args.output_dir/'cache/views.npz',allow_pickle=False)as depth_archive,np.load(args.output_dir/'cache/shaded_views.npz',allow_pickle=False)as surface_archive,np.load(args.output_dir/'cache/text.npz',allow_pickle=False)as text_file:
        depth={key:depth_archive[key]for key in depth_archive.files}
        surface={key:surface_archive[key]for key in surface_archive.files}
        text={key:text_file[key]for key in text_file.files}
        if list(depth['node_indices'])!=[i for i,n in nodes]or list(surface['node_indices'])!=[i for i,n in nodes]:
            raise ValueError('Cannot combine partial or mismatched scene feature caches')
        rows=[];depth_rows=[];surface_rows=[];replay_rows=[]
        for order,(index,node)in enumerate(nodes,start=1):
            dl,dh=depth['offsets'][order-1:order+1];sl,sh=surface['offsets'][order-1:order+1]
            d=infer_evidence(depth['features'][dl:dh],depth['qualities'][dl:dh],depth['kinds'][dl:dh],text,taxonomy,config)
            s=infer_evidence(surface['features'][sl:sh],surface['qualities'][sl:sh],np.zeros(sh-sl,dtype=np.int8),text,taxonomy,config)
            base={'order':order,'node_index':index,'mesh_index':node['mesh'],'node_name':node.get('name',''),
                  'mesh_name':gltf['meshes'][node['mesh']].get('name','')}
            combined=combine(d,s)
            rows.append(serialize_evidence(base,combined));depth_rows.append(serialize_evidence(base,d));surface_rows.append(serialize_evidence(base,s))
            # Cheap cached-feature controls isolate aggregation/orientation from rendering changes.
            features=depth['features'][dl:dh].astype(np.float32)
            kinds=depth['kinds'][dl:dh]
            world=kinds==0
            world_result=infer_evidence(features[world],depth['qualities'][dl:dh][world],np.zeros(int(world.sum()),dtype=np.int8),text,taxonomy,config)
            if np.any(world):
                average=features[world].mean(0);average/=max(np.linalg.norm(average),1e-8)
                flat_id=taxonomy['labels'][int(np.argmax(average@text['labels'].T))]['id']
            else:flat_id=''
            replay_rows.append({**base,'same_input_uniform_flat_candidate':flat_id,
                'world_hierarchical_candidate':world_result.get('candidate_id',''),
                'world_hierarchical_status':world_result['status'],'depth_multiscale_candidate':d.get('candidate_id',''),
                'depth_status':d['status'],'surface_candidate':s.get('candidate_id',''),'surface_status':s['status'],
                'combined_status':combined['status'],'render_agreement':combined['render_agreement'],
                'scope':'Cached-feature controls; not an isolated vocabulary ablation because legacy inputs differ.'})
    for name,data in [('open_semantic_combined.csv',rows),('open_semantic_depth_only.csv',depth_rows),
                     ('open_semantic_surface_only.csv',surface_rows),('cached_feature_controls.csv',replay_rows)]:
        write_rows(args.output_dir/name,data)
    summary={'rows':len(rows),'combined_status':dict(Counter(r['open_status']for r in rows)),
        'depth_status':dict(Counter(r['open_status']for r in depth_rows)),
        'surface_status':dict(Counter(r['open_status']for r in surface_rows)),
        'render_agreement':dict(Counter(r['render_agreement']for r in replay_rows)),
        'accuracy_measured':False,'cad_names_used':False}
    (args.output_dir/'combined_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    inputs=[args.glb,directory/'industrial_open_vocab_v2.json',directory/'open_semantic_v2_config.json',
        directory/'open_semantic_evidence.py',Path(__file__),args.output_dir/'cache/views.npz',
        args.output_dir/'cache/shaded_views.npz',args.output_dir/'cache/text.npz']
    outputs=[args.output_dir/p for p in ['open_semantic_combined.csv','open_semantic_depth_only.csv',
        'open_semantic_surface_only.csv','cached_feature_controls.csv','combined_summary.json']]
    record(args.output_dir/'combined_prediction.provenance.json',stage_identity(inputs,{'single_stream_min_reliability':.65,'single_stream_min_margin':.01,
        'partial_group_min_margin':.01,'partial_group_min_agreement':.5}),outputs)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
