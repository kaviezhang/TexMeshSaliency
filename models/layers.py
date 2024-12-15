import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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


class FaceKernelCorrelation(nn.Module):

    def __init__(self, num_kernel=64, sigma=0.2):
        super(FaceKernelCorrelation, self).__init__()
        self.num_kernel = num_kernel
        self.sigma = sigma
        self.weight_alpha = Parameter(torch.rand(1, num_kernel, 4) * np.pi)
        self.weight_beta = Parameter(torch.rand(1, num_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(num_kernel)
        self.relu = nn.ReLU()

    def forward(self, normals, neighbor_index):

        b, _, n = normals.size()

        center = normals.unsqueeze(2).expand(-1, -1, self.num_kernel, -1).unsqueeze(4)
        neighbor = torch.gather(normals.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                neighbor_index.unsqueeze(1).expand(-1, 3, -1, -1))
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.num_kernel, -1, -1)

        fea = torch.cat([center, neighbor], 4)
        fea = fea.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4)
        weight = torch.cat([torch.sin(self.weight_alpha) * torch.cos(self.weight_beta),
                            torch.sin(self.weight_alpha) * torch.sin(self.weight_beta),
                            torch.cos(self.weight_alpha)], 0)
        weight = weight.unsqueeze(0).expand(b, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, n, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, 4, -1)

        dist = torch.sum((fea - weight)**2, 1)
        fea = torch.sum(torch.sum(np.e**(dist / (-2 * self.sigma**2)), 4), 3) / 16

        return self.relu(self.bn(fea))


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


class StructuralDescriptor(nn.Module):

    def __init__(self, cfg):
        super(StructuralDescriptor, self).__init__()

        self.FRC = FaceRotateConvolution()
        self.FKC = FaceKernelCorrelation(cfg['num_kernel'], cfg['sigma'])
        self.structural_mlp = nn.Sequential(
            nn.Conv1d(64 + 3 + cfg['num_kernel'], 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
        )

    def forward(self, corners, normals, neighbor_index):
        structural_fea1 = self.FRC(corners)
        structural_fea2 = self.FKC(normals, neighbor_index)

        return self.structural_mlp(torch.cat([structural_fea1, structural_fea2, normals], 1))


class MeshConvolution(nn.Module):

    def __init__(self, cfg, spatial_in_channel, structural_in_channel, spatial_out_channel, structural_out_channel):
        super(MeshConvolution, self).__init__()

        self.spatial_in_channel = spatial_in_channel
        self.structural_in_channel = structural_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_out_channel = structural_out_channel

        assert cfg['aggregation_method'] in ['Concat', 'Max', 'Average']
        self.aggregation_method = cfg['aggregation_method']

        self.combination_mlp = nn.Sequential(
            nn.Conv1d(self.spatial_in_channel + self.structural_in_channel, self.spatial_out_channel, 1),
            nn.BatchNorm1d(self.spatial_out_channel),
            nn.ReLU(),
        )

        if self.aggregation_method == 'Concat':
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in_channel * 2, self.structural_in_channel, 1),
                nn.BatchNorm2d(self.structural_in_channel),
                nn.ReLU(),
            )

        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.structural_in_channel, self.structural_out_channel, 1),
            nn.BatchNorm1d(self.structural_out_channel),
            nn.ReLU(),
        )

    def forward(self, spatial_fea, structural_fea, neighbor_index):
        b, _, n = spatial_fea.size()

        # Combination
        spatial_fea = self.combination_mlp(torch.cat([spatial_fea, structural_fea], 1))

        # Aggregation
        if self.aggregation_method == 'Concat':
            structural_fea = torch.cat([structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                                        torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                                     neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel,
                                                                                        -1, -1))], 1)
            structural_fea = self.concat_mlp(structural_fea)
            structural_fea = torch.max(structural_fea, 3)[0]

        elif self.aggregation_method == 'Max':
            structural_fea = torch.cat([structural_fea.unsqueeze(3),
                                        torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                                     neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel,
                                                                                        -1, -1))], 3)
            structural_fea = torch.max(structural_fea, 3)[0]

        elif self.aggregation_method == 'Average':
            structural_fea = torch.cat([structural_fea.unsqueeze(3),
                                        torch.gather(structural_fea.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                                     neighbor_index.unsqueeze(1).expand(-1, self.structural_in_channel,
                                                                                        -1, -1))], 3)
            structural_fea = torch.sum(structural_fea, dim=3) / 4

        structural_fea = self.aggregation_mlp(structural_fea)

        return spatial_fea, structural_fea


class MaxPoolFaceFeature(nn.Module):
    r"""
    Retrives maximum channel value from amonng the faces and its n-Ring neighborhood.
    E.g: Let face "f" and its 1-ring neighbors "n1", "n2", and, "n3" have channels
    "cf", "cn1", "cn2", "cn3" as shown below.
             _ _          _ _          _ _
            |   |        |   |        |   |
          _ |cn1|_       |cf |      _ |cn2| _
          \ |_ _| /      |_ _|      \ |_ _| /
           \ n1  /      /  f  \      \ n2  /
            \   /      /_ _ _ _\      \   /
             \ /          _ _          \ /
                         |   |
                       _ |cn3| _
                       \ |_ _| /
                        \ n3  /
                         \   /
                          \ /

    Then, MaxPoolFaceFeature retrives max(cf, cn1, cn2, cn3) for f and re-assigns it to f.
    """
    def __init__(self, in_channel, num_neighbor=3):
        """
        Args:
            in_channel: number of channels in feature

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(MaxPoolFaceFeature, self).__init__()
        self.in_channel = in_channel
        self.num_neighbor = num_neighbor

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood.
            [num_meshes, num_faces, num_neighbor]

        Returns:
            max_fea: maximum channel value from amonng the faces and its n-Ring neighborhood.
            [num_meshes, in_channel, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert ring_n.shape == (num_meshes, num_faces, self.num_neighbor)
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)

        # Gather features at face neighbors
        fea = fea.unsqueeze(3)
        ring_n = ring_n.unsqueeze(1)
        ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)

        neighbor_fea = fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_channels)[None, :, None, None],
            ring_n
        ]

        neighbor_fea = neighbor_fea.squeeze(4)
        # Concatenate gathered neighbor features to face_feature, and then find the max
        fea = torch.cat([fea, neighbor_fea], 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces, self.num_neighbor + 1)

        max_fea = torch.max(fea, dim=3).values
        # assert max_fea.shape == (num_meshes, self.in_channel, num_faces)

        return max_fea

class ConvFace(nn.Module):
    r"""
    Convolves the channel values of the faces with its n-Ring neighborhood.
    E.g: Let face "f" and its 1-ring neighbors "n1", "n2", "n3" have channels "cf",
    "cn1", "cn2", "cn3" as shown below.
             _ _          _ _          _ _
            |   |        |   |        |   |
          _ |cn1|_       |cf |      _ |cn2| _
          \ |_ _| /      |_ _|      \ |_ _| /
           \ n1  /      /  f  \      \ n2  /
            \   /      /_ _ _ _\      \   /
             \ /          _ _          \ /
                         |   |
                       _ |cn3| _
                       \ |_ _| /
                        \ n3  /
                         \   /
                          \ /

    Then, for f, ConvFace computes sum(cf, cn1, cn2, cn3) and passes it along to
    Conv1D to perform convolution.
    """
    def __init__(self, in_channel, out_channel, num_neighbor):
        """
        Args:
            in_channel: number of channels in feature

            out_channel: number of channels produced by convolution

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(ConvFace, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_neighbor = num_neighbor
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(self.in_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(),
        )

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            conv_fea: features produced by convolution of faces with its
            n-Ring neighborhood features
            [num_meshes, out_channel, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert ring_n.shape == (num_meshes, num_faces, self.num_neighbor)
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)

        # Gather features at face neighbors
        fea = fea.unsqueeze(3)
        ring_n = ring_n.unsqueeze(1)
        ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)

        neighbor_fea = fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_channels)[None, :, None, None],
            ring_n
        ]

        neighbor_fea = neighbor_fea.squeeze(4)
        # Concatenate gathered neighbor features to face_feature, and then find the sum
        fea = torch.cat([fea, neighbor_fea], 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces, self.num_neighbor + 1)

        fea = torch.sum(neighbor_fea, 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces)

        # Convolve
        conv_fea = self.concat_mlp(fea)
        # assert conv_fea.shape == (num_meshes, self.out_channel, num_faces)

        return conv_fea

class ConvFaceBlock(nn.Module):
    """
    Multiple ConvFaceBlock layers create a MeshBlock.
    ConvFaceBlock is comprised of ConvFace layers.
    First ConvFace layer convolves on in_channel to produce "128" channels.
    Second ConvFace convolves these "128" channels to produce "growth factor" channels.
    These features get concatenated to the original input feature to produce
    "in_channel + growth_factor" channels.
    """
    def __init__(self, in_channel, growth_factor, num_neighbor):
        """
        Args:
        in_channel: number of channels in feature

        growth_factor: number of channels to increase in_channel by

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(ConvFaceBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_neighbor = num_neighbor
        self.conv_face_1 = ConvFace(self.in_channel, 128, self.num_neighbor)
        self.conv_face_2 = ConvFace(128, self.growth_factor, self.num_neighbor)

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            conv_block_fea: features produced by ConvFaceBlock layer
            [num_meshes, in_channel + growth_factor, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)
        fea_copy = fea
        fea = self.conv_face_1(fea, ring_n)
        fea = self.conv_face_2(fea, ring_n)
        conv_block_fea = torch.cat([fea_copy, fea], 1)
        # assert conv_block_fea.shape == (num_meshes, self.in_channel + self.growth_factor, num_faces)

        return conv_block_fea

class MeshBlock(nn.ModuleDict):
    """
    Multiple MeshBlock layers create MeshNet2.
    MeshBlock is comprised of several ConvFaceBlock layers.
    """
    def __init__(self, in_channel, num_neighbor, num_block, growth_factor):
        """
        in_channel: number of channels in feature

        growth_factor: number of channels a single ConvFaceBlock increase in_channel by

        num_block: number of ConvFaceBlock layers in a single MeshBlock

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(MeshBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_block = num_block
        self.num_neighbor = num_neighbor

        for i in range(0, num_block):
            layer = ConvFaceBlock(in_channel=in_channel,
                                  growth_factor=growth_factor,
                                  num_neighbor=num_neighbor)
            in_channel += growth_factor
            self.add_module('meshblock%d' % (i + 1), layer)

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            fea: features produced by MeshBlock layer
            [num_meshes, in_channel + growth_factor * num_block, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)
        for _, layer in self.items():
            fea = layer(fea, ring_n)
        # out_channel = self.in_channel + self.growth_factor * self.num_block
        # assert fea.shape == (num_meshes, out_channel, num_faces)
        return fea
