"""Audit empty GLB mesh references without fabricating neural predictions."""


def align_mesh_population(data, supervised, opened):
    refs=[(i,n) for i,n in enumerate(data['nodes']) if 'mesh' in n]
    expected={(str(i),str(n['mesh'])) for i,n in refs}
    excluded=[]
    for order,(i,node) in enumerate(refs,1):
        mesh=data['meshes'][node['mesh']]
        vertices=sum(int(data['accessors'][p['attributes']['POSITION']]['count'])
            for p in mesh.get('primitives',[]) if 'POSITION' in p.get('attributes',{}))
        if not vertices:
            excluded.append(dict(order=order,node_index=i,mesh_index=node['mesh'],
                node_name=node.get('name',f'node_{i}'),reason='no_POSITION_vertices',vertex_count=0,source_faces=0))
    empty={(str(r['node_index']),str(r['mesh_index'])) for r in excluded}
    keys=lambda rows:{(str(r['node_index']),str(r['mesh_index'])) for r in rows}
    assert len(supervised)==len(keys(supervised)) and keys(supervised)==expected-empty, 'Missing or unexpected supervised geometry rows'
    assert len(opened)==len(keys(opened)) and keys(opened)==expected, 'Open evidence does not cover all mesh references'
    assert all(r['open_status']=='abstain' for r in opened if (str(r['node_index']),str(r['mesh_index'])) in empty), 'Empty geometry cannot supply accepted visual evidence'
    valid_open=[r for r in opened if (str(r['node_index']),str(r['mesh_index'])) not in empty]
    return supervised,valid_open,excluded
