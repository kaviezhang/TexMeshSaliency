import numpy as np
import os
import torch
import torch.utils.data as data
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
from pytorch3d.transforms import Rotate, RotateAxisAngle, Transform3d
import math
from torchvision import transforms
import torch.nn.functional as F


class MeshDataset(data.Dataset):

    def __init__(self, cfg, part='train', mesh_paths=None):
        self.augment_data = cfg['augment_data']
        if self.augment_data and part == "train":
            self.augment_vert = False
            self.augment_tex = True
            self.augment_rotation = True
        if part == "test":
            self.augment_vert = False
            self.augment_tex = False
            self.augment_rotation = False

        self.device = torch.device('cpu:0')
        self.root = cfg['data_root']
        self.max_faces = cfg['max_faces']
        self.part = part
        if self.augment_data:
            self.jitter_sigma = cfg['jitter_sigma']
            self.jitter_clip = cfg['jitter_clip']

        self.data = []
        for mesh_path in mesh_paths:
            mesh_name = mesh_path.split("/")[-1].split(".")[0]
            filename = mesh_path
            if filename.endswith('.npz') or filename.endswith('.obj'):
                target_name = os.path.join(self.root+"/SaliencyMap/non_texture", mesh_name+".csv")
                npz_name = os.path.join(self.root+"/Ring/rgb_texture", mesh_name+".npz")
                fix_name = os.path.join(self.root+"/FixationMap", mesh_name+".csv")
                self.data.append((filename, target_name, npz_name, fix_name, mesh_name))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(1024),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=10)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, i):
        filename, target_name, npz_name, fix_name, mesh_name = self.data[i]
        target = np.genfromtxt(target_name, delimiter=',', dtype=float)
        target = torch.from_numpy(target).float()
        ringn = np.load(npz_name)
        mesh = self.collect_data(filename)

        # position data
        verts = mesh['verts']
        centers = mesh['centers']
        normals = mesh['normals']
        corners = mesh['corners']
        # neighbor data
        faces = ringn['faces']
        ring_1 = ringn['ring_1']
        ring_2 = ringn['ring_2']
        ring_3 = ringn['ring_3']
        neighbors = ringn['neighbors']
        face_colors = ringn['face_colors']
        face_textures = ringn['face_textures']
        texture = ringn['texture']
        uv_grid = ringn['uv_grid']

        # augment the texture image
        if self.augment_tex and self.part == 'train':
            texture = np.transpose(texture, axes=(1, 2, 0))
            texture = self.transform(texture)
            texture = texture.numpy()

        # mesh visualization
        if "visual" == "no":
            img_tensor = np.transpose(texture, axes=(1, 2, 0))
            img_tensor = self.transform(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)
            grids = torch.tensor(uv_grid).unsqueeze(0)
            face_textures_ = F.grid_sample(img_tensor, grid=grids, mode='nearest', align_corners=True)
            face_textures_ = face_textures_.squeeze().transpose(0, 1)
            import trimesh
            face_colors_ = face_textures_[:, :, 0, 0]
            mesh_new = trimesh.Trimesh(vertices=verts, faces=faces, face_colors=face_colors_)
            mesh_new.visual.face_colors = face_colors_
            mesh_new.export("visualization/{}.obj".format(mesh_name))

        # Convert to tensor
        faces = torch.from_numpy(faces).long()
        ring_1 = torch.from_numpy(ring_1).long()
        ring_2 = torch.from_numpy(ring_2).long()
        ring_3 = torch.from_numpy(ring_3).long()
        neighbors = torch.from_numpy(neighbors).long()
        face_colors = torch.from_numpy(face_colors).float()
        face_textures = torch.from_numpy(face_textures).float()
        verts = verts.float()
        centers = centers.float()
        normals = normals.float()
        corners = corners.float()

        # get corner vectors
        corners = corners - torch.cat([centers, centers, centers], 1)

        # data augmentation
        if self.augment_data and self.part == 'train':
            # jitter 中心点坐标加噪
            jittered_data = torch.clip(self.jitter_sigma * torch.randn(*face_textures.shape),
                                       -1 * self.jitter_clip, self.jitter_clip)  # clip截取区间值
            face_textures = face_textures + jittered_data

        # Dictionary for collate_batched_meshes
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'centers': centers,
            'normals': normals,
            'corners': corners,
            'neighbors': neighbors,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'target': target,
            'face_colors': face_colors,
            'face_textures': face_textures,
            'texture': texture,
            'uv_grid': uv_grid,
            'mesh_name': mesh_name
        }
        return collated_dict

    def __len__(self):
        return len(self.data)

    def get_random_rotation_matrix(self):
        # 随机生成欧拉角
        theta_x = torch.rand(1).item() * 2 * math.pi  # 0 到 2π 的随机角度
        theta_y = torch.rand(1).item() * 2 * math.pi  # 0 到 2π 的随机角度
        theta_z = torch.rand(1).item() * 2 * math.pi  # 0 到 2π 的随机角度

        # 生成绕X轴的旋转矩阵
        R_x = torch.tensor([
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)]
        ], dtype=torch.float32)

        # 生成绕Y轴的旋转矩阵
        R_y = torch.tensor([
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)]
        ], dtype=torch.float32)

        # 生成绕Z轴的旋转矩阵
        R_z = torch.tensor([
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # 合成最终的旋转矩阵
        R = torch.matmul(torch.matmul(R_z, R_y), R_x)
        return R

    def is_mesh_valid(self, mesh):
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

    def normalize_mesh(self, verts, faces):
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

    def pytorch3D_mesh(self, f_path, device):
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
        # 应用随机旋转变换到网格
        angel = self.get_random_rotation_matrix()
        R = Rotate(angel, device=device)
        if self.augment_rotation and self.part == "train":
            mesh = mesh.update_padded(R.transform_points(mesh.verts_padded()))
        faces = mesh.faces_packed()
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        v_normals = mesh.verts_normals_packed()
        f_normals = mesh.faces_normals_packed()
        # data augmentation
        if self.augment_vert and self.part == 'train':
            # jitter 中心点坐标加噪
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*verts.shape),
                                    -1 * self.jitter_clip, self.jitter_clip)  # clip截取区间值
            verts = verts + jittered_data
        return mesh, faces, verts, edges, v_normals, f_normals

    def fpath(self, dir_name):
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

    def find_neighbor(self, faces, faces_contain_this_vertex, vf1, vf2, except_face):
        for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
            if i != except_face:
                face = faces[i].tolist()
                face.remove(vf1)
                face.remove(vf2)
                return i
        return except_face

    def collect_data(self, path):
        mesh, faces, verts, edges, v_normals, f_normals = self.pytorch3D_mesh(path, self.device)
        max_faces = faces.shape[0]
        if not self.is_mesh_valid(mesh):
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

        # get corners
        corners = verts[faces.long()]
        # Each face is connected to 3 other faces in the 1st Ring
        assert corners.shape == (max_faces, 3, 3)

        centers = torch.sum(corners, axis=1) / 3
        assert centers.shape == (max_faces, 3)

        corners = corners.reshape(-1, 9)
        assert f_normals.shape == (max_faces, 3)

        faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
        assert faces_feature.shape == (max_faces, 15)

        max_ver = 20500
        verts = np.pad(verts, ((0, max_ver - verts.shape[0]), (0, 0)), mode='constant')
        verts = torch.from_numpy(verts)

        collated_dict = {
            'faces': faces,
            'verts': verts,
            'centers': centers,
            'normals': f_normals,
            'corners': corners,
        }
        return collated_dict
