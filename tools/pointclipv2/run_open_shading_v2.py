"""Cache an additional shaded surface stream independently of depth features."""
import argparse
import json
import mmap
import sys
from pathlib import Path
import numpy as np

from run_pointclipv2_zeroshot_glb import read_glb_chunks,mesh_for_node
from run_open_semantic_v2 import image_batch_encode,prepare_images
from open_semantic_shading import sample_oriented_surface,shaded_views
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/pointnext_lod'))
from artifact_cache import stage_identity,matches,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glb',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    root=Path(__file__).parent/'PointCLIP_V2/zeroshot_cls'
    sys.path.insert(0,str(root/'Dassl3D'));sys.path.insert(0,str(root))
    import torch
    from clip import clip
    from trainers.mv_utils_zs import Realistic_Projection
    cache=args.output_dir/'cache/shaded_views.npz'
    cache.parent.mkdir(parents=True,exist_ok=True)
    stamp=cache.with_suffix('.provenance.json')
    sources=[args.glb,Path(__file__),Path(__file__).with_name('open_semantic_shading.py'),
        Path(__file__).with_name('run_open_semantic_v2.py'),Path(__file__).with_name('run_pointclipv2_zeroshot_glb.py'),
        ROOT/'tools/pointnext_lod/gltf_hierarchy.py',root/'trainers/mv_utils_zs.py',root/'clip',Path.home()/'.cache/clip/ViT-B-16.pt']
    identity=stage_identity(sources,{'surface_points':16384,'views':10,'seed':2026,'preprocess':'clip-rgb','limit':args.limit})
    if matches(stamp,identity,[cache]):
        print('REUSE verified shaded views',flush=True)
        return
    model,_=clip.load('ViT-B/16',device='cuda');model.eval()
    projection=Realistic_Projection()
    rotations=projection.rot_mat@projection.rot_mat2
    gltf,offset,_=read_glb_chunks(args.glb)
    nodes=[(i,n)for i,n in enumerate(gltf['nodes'])if 'mesh'in n]
    if args.limit is not None:nodes=nodes[:args.limit]
    features,qualities,offsets,errors=[],[],[0],[]
    pending=[]
    def flush():
        if not pending:return
        p=torch.from_numpy(np.stack([item[1]for item in pending])).cuda()
        n=torch.from_numpy(np.stack([item[2]for item in pending])).cuda()
        with torch.inference_mode():
            raw=shaded_views(p,n,rotations)
            images,q=prepare_images(raw,'clip-rgb')
            encoded=image_batch_encode(model,images,32)
        for j,(index,_,_)in enumerate(pending):
            features[index]=encoded[j*10:(j+1)*10]
            qualities[index]=q[j*10:(j+1)*10]
        pending.clear()
    with args.glb.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as mm:
        for j,(index,node)in enumerate(nodes):
            features.append(np.empty((0,512),dtype=np.float16));qualities.append(np.empty(0,dtype=np.float32))
            try:
                vertices,faces=mesh_for_node(gltf,mm,offset,node)
                p,n=sample_oriented_surface(vertices,faces,16384,2026+node['mesh'])
                pending.append((j,p,n))
            except Exception as exc:errors.append({'node_index':index,'error':str(exc)})
            if len(pending)>=8:flush()
            if (j+1)%250==0:print(f'Shaded views {j+1}/{len(nodes)}',flush=True)
        flush()
    for array in features:offsets.append(offsets[-1]+len(array))
    np.savez(cache,features=np.concatenate(features),qualities=np.concatenate(qualities),offsets=np.array(offsets),
             node_indices=np.array([i for i,n in nodes]),kinds=np.full(offsets[-1],3,dtype=np.int8))
    record(stamp,identity,[cache])
    (args.output_dir/'shading_status.json').write_text(json.dumps({'status':'complete','rows':len(nodes),'failed_geometry':errors,
        'views_encoded':offsets[-1],'partial_smoke_test':args.limit is not None,'scope':'Surface-normal shading, no semantic metadata.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Shading complete',len(nodes),'nodes',len(errors),'errors',flush=True)


if __name__=='__main__':main()
