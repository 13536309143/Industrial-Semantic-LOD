"""Surface-normal shaded orthographic views complement coarse depth projection."""
import numpy as np


def sample_oriented_surface(vertices, faces, count, seed):
    rng = np.random.default_rng(seed)
    if not len(faces):
        raise ValueError('Surface rendering requires triangles')
    triangles = vertices[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-12)
    triangles, normals, lengths = triangles[valid], normals[valid], lengths[valid]
    if not len(triangles):
        raise ValueError('No nondegenerate surface triangles')
    indices = rng.choice(len(triangles), count, p=lengths/lengths.sum())
    selected = triangles[indices]
    u, v = np.sqrt(rng.random((count, 1))), rng.random((count, 1))
    points = (1-u)*selected[:, 0]+u*(1-v)*selected[:, 1]+u*v*selected[:, 2]
    normals = normals[indices]/lengths[indices, None]
    centered = points-points.mean(0)
    values, basis = np.linalg.eigh(centered.T@centered/count)
    basis = basis[:, np.argsort(values)[::-1]]
    for axis in range(2):
        coordinate = centered@basis[:, axis]
        if coordinate[np.argmax(np.abs(coordinate))] < 0:
            basis[:, axis] *= -1
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    points, normals = centered@basis, normals@basis
    points -= (points.max(0)+points.min(0))*.5
    radius = np.linalg.norm(points, axis=1).max()
    if radius <= 1e-12:
        raise ValueError('Degenerate surface extent')
    return (points/radius).astype(np.float32), normals.astype(np.float32)


def shaded_views(points, normals, rotations, resolution=224):
    import torch
    import torch.nn.functional as F
    batch, count, _ = points.shape
    views = len(rotations)
    xyz = torch.einsum('bni,vij->bvnj', points, rotations).reshape(batch*views, count, 3)
    normal = torch.einsum('bni,vij->bvnj', normals, rotations).reshape(batch*views, count, 3)
    pixel = (((xyz[..., :2]*.86+1)*.5)*(resolution-1)).long().clamp(1, resolution-2)
    index = pixel[..., 1]*resolution+pixel[..., 0]
    zbuffer = torch.full((batch*views, resolution*resolution), float('inf'), device=points.device)
    zbuffer.scatter_reduce_(1,index,xyz[..., 2],reduce='amin',include_self=True)
    visible = (xyz[..., 2]-zbuffer.gather(1,index) <= .025).float()
    light = points.new_tensor([.35,-.25,-.9]); light /= light.norm()
    luminance = .25+.70*torch.abs((normal*light).sum(-1)).clamp(0,1)
    # A neutral CAD material; no class/name/material-color information is used.
    color = luminance[..., None]*points.new_tensor([.70,.75,.80])
    sums = torch.zeros((batch*views,3,resolution*resolution),device=points.device)
    sums.scatter_add_(2,index[:,None,:].expand(-1,3,-1),(color*visible[...,None]).transpose(1,2))
    weights = torch.zeros_like(zbuffer)
    weights.scatter_add_(1,index,visible)
    sums=sums.reshape(-1,3,resolution,resolution)
    weights=weights.reshape(-1,1,resolution,resolution)
    # Small kernel fills point-sampling gaps while retaining larger openings.
    sums=F.avg_pool2d(sums,3,stride=1,padding=1)
    weights=F.avg_pool2d(weights,3,stride=1,padding=1)
    return torch.where(weights>0,sums/weights.clamp_min(1e-8),torch.ones_like(sums)).clamp(0,1)
