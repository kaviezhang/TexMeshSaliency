"""
Data is pre-processed to obtain the following infomation:
    1) Vertices and Faces of the mesh
    2) 1 Ring, 2 Ring, and 3 Ring neighborhood of the mesh faces
"""
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
import shutil
import trimesh
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
# import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F


def is_mesh_valid(mesh):
    """
    Check validity of pytorch3D mesh

    Args:
        mesh: pytorch3D mesh

    Returns:
        validity: validity of the mesh
    """
    validity = True

    # Check if the mesh is not empty
    if mesh.isempty():
        validity = False

    # Check if vertices in the mesh are valid
    verts = mesh.verts_packed()
    if not torch.isfinite(verts).all() or torch.isnan(verts).all():
        validity = False

    # Check if vertex normals in the mesh are valid
    v_normals = mesh.verts_normals_packed()
    if not torch.isfinite(v_normals).all() or torch.isnan(v_normals).all():
        validity = False

    # Check if face normals in the mesh are valid
    f_normals = mesh.faces_normals_packed()
    if not torch.isfinite(f_normals).all() or torch.isnan(f_normals).all():
        validity = False

    return validity


def normalize_mesh(verts, faces):
    """
    Normalize and center input mesh to fit in a sphere of radius 1 centered at (0,0,0)

    Args:
        mesh: pytorch3D mesh

    Returns:
        mesh, faces, verts, edges, v_normals, f_normals: normalized pytorch3D mesh and other mesh
        information
    """
    verts = verts - verts.mean(0)
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    mesh = Meshes(verts=[verts], faces=[faces])
    faces = mesh.faces_packed().squeeze(0)
    verts = mesh.verts_packed().squeeze(0)
    edges = mesh.edges_packed().squeeze(0)
    v_normals = mesh.verts_normals_packed().squeeze(0)
    f_normals = mesh.faces_normals_packed().squeeze(0)

    return mesh, faces, verts, edges, v_normals, f_normals


def pytorch3D_mesh(f_path, device):
    """
    Read pytorch3D mesh from path

    Args:
        f_path: obj file path

    Returns:
        mesh, faces, verts, edges, v_normals, f_normals: pytorch3D mesh and other mesh information
    """
    if not f_path.endswith('.obj'):
        raise ValueError('Input files should be in obj format.')
    mesh = load_objs_as_meshes([f_path], device)
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    v_normals = mesh.verts_normals_packed()
    f_normals = mesh.faces_normals_packed()

    # trimesh load uv and texture
    # transform = transforms.ToTensor()
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    # ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    vertices, faces_idx, aux = load_obj(f_path)
    grid = 2.0 * aux.verts_uvs[faces_idx.textures_idx] - 1
    grid = grid.unsqueeze(0)
    img = aux.texture_images['material_0'].numpy()
    img_tensor = transform(img).unsqueeze(0).flip(2)

    # Grid Sample face colors
    face_colors = F.grid_sample(img_tensor, grid=grid, mode='nearest', align_corners=True)
    face_colors = face_colors.squeeze().permute(1, 2, 0)

    # Grid Sample Texture Patches
    grid_size = 9
    x_min = torch.min(grid[:, :, :, 0], dim=2)[0]
    x_max = torch.max(grid[:, :, :, 0], dim=2)[0]
    y_min = torch.min(grid[:, :, :, 1], dim=2)[0]
    y_max = torch.max(grid[:, :, :, 1], dim=2)[0]
    step_x = (x_max - x_min) / (grid_size - 1)
    step_y = (y_max - y_min) / (grid_size - 1)
    indices = torch.linspace(0, grid_size - 1, grid_size)

    # get min/max step and min_edge
    aspect_ratio = 'keep_aspect_ratio'  # 'keep_aspect_ratio' or 'resize_to_square'
    if aspect_ratio == 'keep_aspect_ratio':
        min_max = torch.cat([x_max - x_min, y_max - y_min], dim=0).min(0)[1]
        x_min[0, min_max == 0] = x_min[0, min_max == 0] - 0.5 * ((y_max - y_min) - (x_max - x_min))[0, min_max == 0]
        y_min[0, min_max == 1] = y_min[0, min_max == 1] + 0.5 * ((y_max - y_min) - (x_max - x_min))[0, min_max == 1]
        step_x[0, min_max == 0] = step_y[0, min_max == 0]
        step_y[0, min_max == 1] = step_x[0, min_max == 1]

    xx_linspaces = x_min[:, :, None] + indices * step_x[:, :, None]
    yy_linspaces = y_min[:, :, None] + indices * step_y[:, :, None]
    grid_x = xx_linspaces.unsqueeze(3).expand(-1, -1, -1, grid_size)
    grid_y = yy_linspaces.unsqueeze(2).expand(-1, -1, grid_size, -1)
    grid_mean = torch.stack([grid_x, grid_y], dim=-1)
    # at 1
    face_textures1 = F.grid_sample(img_tensor.expand(32868, -1, -1, -1),
                                   grid=grid_mean.squeeze(), mode='nearest', align_corners=True)
    # at 2
    img_tensor_ = img_tensor.unsqueeze(2)
    grid_mean_ = torch.cat([grid_mean, torch.zeros_like(grid_mean)[:,:,:,:,0].unsqueeze(-1)], -1)
    face_textures2 = F.grid_sample(img_tensor_, grid=grid_mean_, mode='nearest', align_corners=True)
    face_textures2 = face_textures2.squeeze().transpose(0, 1)

    face_textures = face_textures1
    # face_colors = face_textures[:, :, 0, 0]
    # face_colors = (face_colors - torch.min(face_colors)) / (torch.max(face_colors) - torch.min(face_colors))
    # face_colors = face_textures.reshape((face_textures.shape[0], face_textures.shape[1], -1))
    # face_colors = torch.mean(face_colors, 2)
    # import trimesh
    # mesh_new = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)
    # mesh_new.visual.face_colors = face_colors
    # mesh_new.export("aaa.obj")

    # save texture and uv
    texture = img_tensor.squeeze()
    uv_grid = grid_mean_.squeeze()
    # print(texture.shape)
    return (mesh, faces, verts, edges, v_normals, f_normals,
            face_colors, face_textures, texture, uv_grid)


def fpath(dir_name):
    """
    Return all obj file in a directory

    Args:
        dir_name: root path to obj files

    Returns:
        f_path: list of obj files paths
    """
    f_path = []
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for f in files:
            if f.endswith('.obj'):
                if os.path.exists(os.path.join(root, f)):
                    f_path.append(os.path.join(root, f))
    return f_path


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


device = torch.device('cpu:0')
# To process the dataset enter the path where they are stored
data_root = '/home/kaiwei/Dataset/ColorMesh/MeshFile/rgb_texture'
max_faces = 500
if not os.path.exists(data_root):
    raise Exception('Dataset not found at {0}'.format(data_root))

paths_dataset = []
fpath_data = fpath(data_root)
fpath_data.sort()
for path in fpath_data:
    (mesh, faces, verts, edges, v_normals, f_normals,
     face_colors, face_textures, texture, uv_grid) = pytorch3D_mesh(path, device)
    max_faces = faces.shape[0]
    if faces.shape[0] != 32868:
        continue
    if not is_mesh_valid(mesh):
        raise ValueError('Mesh is invalid!')
    assert faces.shape[0] == (max_faces)

    # Normalize Mesh
    # mesh, faces, verts, edges, v_normals, f_normals = normalize_mesh(verts=verts, faces=faces)

    # move to center
    center = (torch.max(verts, 0)[0] + torch.min(verts, 0)[0]) / 2
    verts -= center

    # normalize
    max_len = torch.max(verts[:, 0] ** 2 + verts[:, 1] ** 2 + verts[:, 2] ** 2)
    verts /= torch.sqrt(max_len)

    # get neighbors
    faces_contain_this_vertex = []
    for i in range(len(verts)):
        faces_contain_this_vertex.append(set([]))
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)
    neighbors = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])
    neighbors = np.array(neighbors)

    ########################################################################### 1st-Ring ###########################################################################
    data = Data(pos=verts, edge_index=edges.permute(1, 0), face=faces.permute(1, 0))
    trimesh = to_trimesh(data)
    # Neighbor faces index along edges, Edges along neighbor_faces
    faces_adjacency, edges_adjacency = face_adjacency(faces=faces.permute(1, 0),
                                                      mesh=trimesh,
                                                      return_edges=True)

    faces_neighbor_1st_ring = []
    edges_neighbor_1ring = []

    # For each face get 1-Ring neighborhood along its edges
    # For each face get edge between face and neighbor faces
    for face_idx in range(max_faces):
        face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
        face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)

        face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
        face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]

        face_neighbor_1st_ring = np.concatenate([face_neighbor_dim_0,
                                                 face_neighbor_dim_1])

        # Edge between face and neighbor faces
        face_edge = np.concatenate([face_dim_0, face_dim_1]).reshape(-1)
        edge_neighbor_1ring = edges_adjacency[face_edge]

        faces_neighbor_1st_ring.insert(face_idx, face_neighbor_1st_ring)
        edges_neighbor_1ring.insert(face_idx, edge_neighbor_1ring)

    try:
        np.asarray(faces_neighbor_1st_ring)
    except:
        print("{} failed".format(path))
        continue
    paths_dataset.append(path)
    faces_neighbor_1st_ring = np.asarray(faces_neighbor_1st_ring).squeeze(2)
    edges_neighbor_1ring = np.asarray(edges_neighbor_1ring)

    # Each face is connected to 3 other faces in the 1st Ring
    assert faces_neighbor_1st_ring.shape == (max_faces, 3)
    # Each face has 1 edge between itself and neighbor faces
    # 2 in last dim since each edge is composed of 2 vertices
    assert edges_neighbor_1ring.shape == (max_faces, 3, 2)

    ########################################################################### 2nd-Ring ###########################################################################
    faces_neighbor_0th_ring = np.arange(max_faces)
    faces_neighbor_2ring = faces_neighbor_1st_ring[faces_neighbor_1st_ring]
    faces_neighbor_0ring = np.stack([faces_neighbor_0th_ring]*3, axis=1)
    faces_neighbor_0ring = np.stack([faces_neighbor_0ring]*3, axis=2)

    dilation_mask = faces_neighbor_2ring != faces_neighbor_0ring
    faces_neighbor_2nd_ring = faces_neighbor_2ring[dilation_mask]
    faces_neighbor_2nd_ring = faces_neighbor_2nd_ring.reshape(max_faces, -1)

    # For each face there are 6 neighboring faces in its 2-Ring neighborhood
    assert faces_neighbor_2nd_ring.shape == (max_faces, 6)

    ########################################################################### 3rd-Ring ###########################################################################
    faces_neighbor_3ring = faces_neighbor_2nd_ring[faces_neighbor_1st_ring]
    faces_neighbor_3ring = faces_neighbor_3ring.reshape(max_faces, -1)

    faces_neighbor_3rd_ring = []
    for face_idx in range(max_faces):
        face_neighbor_3ring = faces_neighbor_3ring[face_idx]
        for neighbor in range(3):
            face_neighbor_1st_ring = faces_neighbor_1st_ring[face_idx, neighbor]
            dilation_mask = np.delete(
                np.arange(face_neighbor_3ring.shape[0]),
                np.where(face_neighbor_3ring == face_neighbor_1st_ring)[0][0:2])
            face_neighbor_3ring = face_neighbor_3ring[dilation_mask]
        faces_neighbor_3rd_ring.insert(face_idx, face_neighbor_3ring)
    # For each face there are 12 neighboring faces in its 3-Ring neighborhood
    faces_neighbor_3rd_ring = np.array(faces_neighbor_3rd_ring)
    assert faces_neighbor_3rd_ring.shape == (max_faces, 12)

    path_ring = path.replace('.obj', '.npz').replace('MeshFile', 'Ring')
    path_ring = "/home/kaiwei/Dataset/ColorMesh/Ring/"+path_ring.split("/")[-3]+"/"+path_ring.split("/")[-1]
    path_sal = path.replace('.obj', '.csv').replace('MeshFile', 'SaliencyMap')
    path_sal = "/home/kaiwei/Dataset/ColorMesh/SaliencyMap/"+path_sal.split("/")[-3]+"/"+path_sal.split("/")[-1]
    target = np.genfromtxt(path_sal, delimiter=',', dtype=float)
    print(path)
    path_dist = path.replace('MeshFile', 'RingMesh')
    # shutil.copy(path, path_dist)

    np.savez(path_ring,
             faces=faces,
             ring_1=faces_neighbor_1st_ring,
             ring_2=faces_neighbor_2nd_ring,
             ring_3=faces_neighbor_3rd_ring,
             target=target,
             neighbors=neighbors,
             face_colors=face_colors,
             face_textures=face_textures,
             texture=texture,
             uv_grid=uv_grid)
paths_dataset = np.array(paths_dataset)
np.savetxt("paths_dataset_rgb.txt", paths_dataset, fmt="%s", delimiter=",")
