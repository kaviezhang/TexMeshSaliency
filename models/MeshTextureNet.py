import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PointDescriptor, NormalDescriptor
from models import ConvSurface
from models import MaxPoolFaceFeature, MeshBlock
from models import PsuedoMeshBlock
from torch.nn.parameter import Parameter
import math
import torchvision.models as models


class MeshTextureNet(nn.Module):
    """ MeshNet++ Model"""
    def __init__(self, cfg):
        """
        Args:
            cfg: configuration file
            num_faces: number of mesh faces
            num_cls: number of classes in dataset
        """
        # Setup
        super(MeshTextureNet, self).__init__()
        self.spatial_descriptor = SpatialDescriptor()
        # self.FRC = FaceRotateConvolution()
        self.FPR = FacePostionRelation()
        # self.FSC = FaceShapeConvolution()
        self.FaceTex_Descriptor = FaceTex_Descriptor()
        # self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        self.curve_descriptor_1 = CurveDescriptor(cfg=cfg['curve_descriptor'], num_neighbor=3)
        self.curve_descriptor_2 = CurveDescriptor(cfg=cfg['curve_descriptor'], num_neighbor=6)
        self.curve_descriptor_3 = CurveDescriptor(cfg=cfg['curve_descriptor'], num_neighbor=12)
        self.curve_fusion = nn.Sequential(
            nn.Conv1d(64+cfg['curve_descriptor']['num_kernel']*3, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
        )

        # self.conv_surface_1 = ConvSurface(num_faces=num_faces, num_neighbor=3, cfg=cfg['ConvSurface'])
        # self.conv_surface_2 = ConvSurface(num_faces=num_faces, num_neighbor=6, cfg=cfg['ConvSurface'])
        # self.conv_surface_3 = ConvSurface(num_faces=num_faces, num_neighbor=12, cfg=cfg['ConvSurface'])
        self.mesh_conv1 = MeshConvolution(cfg['mesh_convolution'],
                                          64,
                                          131,
                                          64,
                                          256,
                                          256)
        self.mesh_conv2 = MeshConvolution(cfg['mesh_convolution'],
                                          256,
                                          256,
                                          64,
                                          512,
                                          512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        in_channel = 1024
        self.conv_out = nn.Sequential(
            nn.Conv1d(in_channel, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

        print('Structural descriptor number of learnable kernels: {0}'.format(cfg['curve_descriptor']['num_kernel']))

    def forward(self, verts, faces, centers, normals, corners, neighbor_index,
                ring_1, ring_2, ring_3, face_colors, face_textures, texture, uv_grid):
        # Face center features
        spatial_fea0 = self.spatial_descriptor(centers)
        # structural_fea0 = self.FRC(corners)
        structural_fea0 = self.FPR(centers, ring_1, corners, verts, faces)
        # structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)
        curve_fea1 = self.curve_descriptor_1(normals, ring_1)
        curve_fea2 = self.curve_descriptor_2(normals, ring_2)
        curve_fea3 = self.curve_descriptor_3(normals, ring_3)

        structural_fea0 = self.curve_fusion(torch.cat([structural_fea0, curve_fea1, curve_fea2, curve_fea3], 1))

        # # Surface features from 1-Ring neighborhood around a face
        # surface_fea_1 = self.conv_surface_1(verts=verts,
        #                                     faces=faces,
        #                                     ring_n=ring_1,
        #                                     centers=centers)
        #
        # # Surface features from 2-Ring neighborhood around a face
        # surface_fea_2 = self.conv_surface_2(verts=verts,
        #                                     faces=faces,
        #                                     ring_n=ring_2,
        #                                     centers=centers)
        #
        # # Surface features from 3-Ring neighborhood around a face
        # surface_fea_3 = self.conv_surface_3(verts=verts,
        #                                     faces=faces,
        #                                     ring_n=ring_3,
        #                                     centers=centers)

        # Face Texture Features
        tex_fea = self.FaceTex_Descriptor(face_textures, texture, uv_grid)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index, tex_fea)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index, tex_fea)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        # Concatenate spatial and structural features
        fea = torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1)
        fea = self.concat_mlp(fea)

        fea = self.conv_out(fea)
        fea = torch.squeeze(fea, 1)
        fea = self.sigmoid(fea)

        return fea


class CurveDescriptor(nn.Module):
    def __init__(self,  cfg, num_neighbor=3):
        super(CurveDescriptor, self).__init__()
        self.num_kernel = cfg['num_kernel']
        self.num_neighbor = num_neighbor
        self.directions = nn.Parameter(torch.FloatTensor(3, self.num_kernel))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(self.num_kernel)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.num_kernel)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, normals, ring_n):
        # take the normals of neighbor faces by index ring_n
        normals = normals.permute(0, 2, 1)
        normals_exp = normals.unsqueeze(2).expand(-1, -1, self.num_neighbor, -1)
        ring_n_exp = ring_n.unsqueeze(3).expand(-1, -1, -1, 3)
        normals_ring = torch.gather(normals_exp, 1, ring_n_exp)
        normals_ring = torch.cat([normals_ring, normals.unsqueeze(2)], 2)
        neighbor_direction_norm = F.normalize(normals_ring, dim=-1)
        support_direction_norm = F.normalize(self.directions, dim=0)
        feature = neighbor_direction_norm @ support_direction_norm
        # assert feature.shape == (num_meshes, num_faces, self.num_samples, self.num_kernel)

        feature = torch.max(feature, dim=2)[0]
        feature = feature.permute(0, 2, 1)
        feature = self.relu(self.bn(feature))
        return feature


class FaceShapeConvolution(nn.Module):
    def __init__(self):
        super(FaceShapeConvolution, self).__init__()

    def forward(self, centers, ring_n, corners, verts, faces):
        return centers


class FacePostionRelation(nn.Module):
    def __init__(self):
        super(FacePostionRelation, self).__init__()
        self.num_neighbor = 3
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, centers, ring_n, corners, verts, faces):
        # take the centers of neighbor faces by index ring_n
        centers = centers.permute(0, 2, 1)
        centers_exp = centers.unsqueeze(2).expand(-1, -1, self.num_neighbor, -1)
        ring_n_exp = ring_n.unsqueeze(3).expand(-1, -1, -1, 3)
        centers_ring = torch.gather(centers_exp, 1, ring_n_exp)
        center_vectors = centers_ring - centers.unsqueeze(3)
        center_vectors = center_vectors.flatten(2, 3)
        center_vectors = center_vectors.permute(0, 2, 1)
        # normals_ring = torch.cat([normals_ring, centers.unsqueeze(2)], 2)

        # vertices gather
        verts_exp = verts.unsqueeze(2).expand(-1, -1, 3, -1)
        faces_exp = faces.unsqueeze(3).expand(-1, -1, -1, 3)
        verts_face = torch.gather(verts_exp, 1, faces_exp)
        verts_face = verts_face.flatten(2, 3)
        verts_face = verts_face.permute(0, 2, 1)



        fea = (self.rotate_mlp(corners[:, :6]) +
               self.rotate_mlp(corners[:, 3:9]) +
               self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3
        return self.fusion_mlp(fea)


class FaceRotateConvolution(nn.Module):

    def __init__(self):
        super(FaceRotateConvolution, self).__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, corners):

        fea = (self.rotate_mlp(corners[:, :6]) +
               self.rotate_mlp(corners[:, 3:9]) +
               self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3

        return self.fusion_mlp(fea)


class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers):
        return self.spatial_mlp(centers)


class Spatial_Descriptor(nn.Module):

    def __init__(self):
        super(Spatial_Descriptor, self).__init__()

        self.centers_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.vertices_mlp = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers, verts, faces):
        # normals = verts.permute(0, 2, 1)
        verts_exp = verts.unsqueeze(2).expand(-1, -1, 3, -1)
        faces_exp = faces.unsqueeze(3).expand(-1, -1, -1, 3)
        corners = torch.gather(verts_exp, 1, faces_exp)
        corners = corners.flatten(2, 3).permute(0, 2, 1)
        corners = (self.vertices_mlp(corners[:, :6]) +
               self.vertices_mlp(corners[:, 3:9]) +
               self.vertices_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3
        return self.spatial_mlp(torch.cat([self.centers_mlp(centers), corners], 1))


class MeshConvolution(nn.Module):

    def __init__(self, cfg, spatial_in_channel, structural_in_channel, tex_in_channel, spatial_out_channel, structural_out_channel):
        super(MeshConvolution, self).__init__()

        self.spatial_in_channel = spatial_in_channel
        self.structural_in_channel = structural_in_channel
        self.tex_in_channel = tex_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_out_channel = structural_out_channel

        assert cfg['aggregation_method'] in ['Concat', 'Max', 'Average', 'Aggregation']
        self.aggregation_method = cfg['aggregation_method']

        self.combination_mlp = nn.Sequential(
            nn.Conv1d(self.spatial_in_channel + self.structural_in_channel + self.tex_in_channel,
                      self.spatial_out_channel, 1),
            nn.BatchNorm1d(self.spatial_out_channel),
            nn.ReLU(),
        )

        if self.aggregation_method == 'Aggregation':
            self.concat_mlp = nn.Sequential(
                nn.Conv1d(self.structural_in_channel * 4, self.structural_in_channel, 1),
                nn.BatchNorm1d(self.structural_in_channel),
                nn.ReLU(),
            )
        if self.aggregation_method == 'Concat':
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in_channel * 2, self.structural_in_channel, 1),
                nn.BatchNorm2d(self.structural_in_channel),
                nn.ReLU(),
            )
        if self.aggregation_method == 'Average':
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in_channel * 1, self.structural_in_channel, 1),
                nn.BatchNorm2d(self.structural_in_channel),
                nn.ReLU(),
            )
        if self.aggregation_method == 'Max':
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in_channel * 1, self.structural_in_channel, 1),
                nn.BatchNorm2d(self.structural_in_channel),
                nn.ReLU(),
            )
        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.structural_in_channel, self.structural_out_channel, 1),
            nn.BatchNorm1d(self.structural_out_channel),
            nn.ReLU(),
        )

    def forward(self, spatial_fea, structural_fea, neighbor_index, tex_fea):
        b, _, n = spatial_fea.size()

        # Combination
        spatial_fea = self.combination_mlp(torch.cat([spatial_fea, structural_fea, tex_fea], 1))

        # Aggregation
        if self.aggregation_method == 'Concat':
            structural_fea = torch.cat([
                structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                             neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))
            ], 1)
            structural_fea = self.concat_mlp(structural_fea)
            structural_fea = torch.max(structural_fea, 3)[0]
        if self.aggregation_method == 'Aggregation':
            structural_fea_n = torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                            neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))
            # sum
            structural_fea_sum = torch.sum(structural_fea_n, 3)
            # sum of diff
            structural_fea_dif = (torch.abs(structural_fea_n[:, :, :, 2] - structural_fea_n[:, :, :, 1]) +
                                  torch.abs(structural_fea_n[:, :, :, 1] - structural_fea_n[:, :, :, 0]) +
                                  torch.abs(structural_fea_n[:, :, :, 0] - structural_fea_n[:, :, :, 1]))
            # sum of div_center
            structural_fea_div = torch.abs(structural_fea_n - structural_fea.unsqueeze(3))
            structural_fea_div = torch.sum(structural_fea_div, 3)
            structural_fea = torch.cat([structural_fea, structural_fea_sum, structural_fea_dif, structural_fea_div], 1)
            structural_fea = self.concat_mlp(structural_fea)
        if self.aggregation_method == 'Average':
            structural_fea = torch.cat([
                structural_fea.unsqueeze(3),
                torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                             neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))
            ], 3)
            structural_fea = torch.mean(structural_fea, 3)
        if self.aggregation_method == 'Max':
            structural_fea = torch.cat([
                structural_fea.unsqueeze(3),
                torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                             neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel, -1, -1))
            ], 3)
            structural_fea = torch.max(structural_fea, dim=3)[0]

        structural_fea = self.aggregation_mlp(structural_fea)

        return spatial_fea, structural_fea


class FaceTex_Descriptor(nn.Module):
    def __init__(self):
        super(FaceTex_Descriptor, self).__init__()
        # backbone = models.resnet34(models.ResNet34_Weights)
        # self.extractor = nn.Sequential(backbone.conv1,
        #                                backbone.bn1,
        #                                backbone.relu,
        #                                backbone.layer1)
        self.texture_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.tex_conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

    def forward(self, face_textures, texture, uv_grid):
        texture = self.texture_conv(texture)
        # texture = self.extractor(texture)
        # grid the sample
        texture = texture.unsqueeze(2)
        face_textures = F.grid_sample(texture, grid=uv_grid, mode='bilinear', align_corners=True)
        face_textures = face_textures.transpose(1, 2)

        B, N, C, H, W = face_textures.shape
        face_textures = face_textures.reshape(B*N, C, H, W)
        fea = self.tex_conv(face_textures)
        fea = fea.squeeze().reshape(B, N, -1)
        fea = fea.permute(0, 2, 1)

        return fea
