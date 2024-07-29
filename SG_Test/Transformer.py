import os
from typing import Dict, Union, List, Tuple

import math
import torch
from torch import nn, Tensor
from torch.quantization import quantize_dynamic
import torch.nn.functional as F

from copy import deepcopy
from typing import Optional
from scipy.spatial import KDTree
import torch
from torch import nn, Tensor
import util

from network import dla60x, load_model, conformer
from collections import OrderedDict

_8_to_16_neighbor=[[0, 3],
                   [1, 6],
                   [2, 7],
                   [3, 8],
                   [9, 12],
                   [13, 10],
                   [14, 11],
                   [15, 12]]

_8_to_32_neighbor=[[0, 5, 3, 8],
                    [9, 6, 1, 14],
                    [10, 15, 7, 2],
                    [16, 11, 8, 24],
                    [17, 12, 25, 20],
                    [18, 26, 13, 21],
                    [27, 19, 24, 22],
                    [28, 31, 30, 25]]


_16_to_32_neighbor=[[0, 3],
                        [1, 6],
                        [2, 7],
                        [3, 11],
                        [4, 12],
                        [13, 5],
                        [14, 6],
                        [15, 20],
                        [16, 21],
                        [22, 17],
                        [23, 18],
                        [24, 19],
                        [25, 20],
                        [26, 29],
                        [27, 30],
                        [31, 28]]

_32_to_64_neighbor=[[ 0,  3],
                        [ 1,  6],
                        [ 2, 10],
                        [ 3, 11],
                        [ 4, 12],
                        [ 5, 13],
                        [ 6, 14],
                        [20, 15],
                        [21, 16],
                        [22, 17],
                        [23, 18],
                        [24, 32],
                        [25, 33],
                        [26, 34],
                        [27, 35],
                        [36, 28],
                        [37, 29],
                        [38, 30],
                        [39, 31],
                        [40, 32],
                        [41, 33],
                        [42, 34],
                        [43, 35],
                        [44, 49],
                        [45, 50],
                        [46, 51],
                        [47, 52],
                        [48, 53],
                        [57, 54],
                        [58, 55],
                        [59, 56],
                        [62, 60]]

_64_to_128_neighbor=[[  0,   3],
                        [  1,   6],
                        [  2,  10],
                        [  3,  11],
                        [  4,  12],
                        [  5,  13],
                        [  6,  19],
                        [ 20,   7],
                        [ 21,   8],
                        [ 22,   9],
                        [ 23,  31],
                        [ 24,  32],
                        [ 25,  33],
                        [ 26,  34],
                        [ 35,  27],
                        [ 36,  28],
                        [ 37,  29],
                        [ 38,  30],
                        [ 39,  31],
                        [ 40,  32],
                        [ 41,  54],
                        [ 42,  55],
                        [ 43,  56],
                        [ 44,  57],
                        [ 45,  58],
                        [ 46,  59],
                        [ 60,  47],
                        [ 61,  48],
                        [ 62,  49],
                        [ 63,  50],
                        [ 64,  51],
                        [ 65,  52],
                        [ 66,  53],
                        [ 67,  54],
                        [ 68,  55],
                        [ 69,  56],
                        [ 70,  78],
                        [ 71,  79],
                        [ 72,  80],
                        [ 73,  81],
                        [ 74,  95],
                        [ 75,  96],
                        [ 76,  97],
                        [ 77,  98],
                        [ 78,  99],
                        [100,  79],
                        [101,  80],
                        [102,  81],
                        [103,  95],
                        [104,  96],
                        [105,  97],
                        [106,  98],
                        [107,  99],
                        [108, 100],
                        [109, 101],
                        [110, 102],
                        [111, 119],
                        [112, 120],
                        [113, 121],
                        [114, 122],
                        [115, 123],
                        [124, 116],
                        [125, 122],
                        [126, 123]]


def neighbor_mask(ln_8,ln_16,radius):
    ln_8 = ln_8    
    anchors_8 = util.sphere_points(ln_8) # anchors.points array that contain coordinates of the points on the surface of a sphere
    anchors_polar_8 = util.cartesian_to_polar(anchors_8) 
    ln_16 = ln_16    
    anchors_16 = util.sphere_points(ln_16) 
    anchors_polar_16 = util.cartesian_to_polar(anchors_16) 
    mask_ln8 = torch.ones((ln_8+1, ln_16+1), dtype=bool)#all true
    mask_ln8[0,0] = False

    # Build a KDTree for point set A
    tree_B = KDTree(anchors_16)

    # Function to find neighbors on left, right, up, and down for each point in B
    def find_neighbors(point_set_A, tree, radius=radius):
        neighbors = []
        for point in point_set_A:
            indices = tree.query_ball_point(point, radius)
            neighbors.append(indices)
        return neighbors

    # Find neighbors for each point in B
    neighbors_A = find_neighbors(anchors_8, tree_B)
    for i, neighbors in enumerate(neighbors_A):
        for j in neighbors:
            mask_ln8[i+1,j+1] = False

    return neighbors_A,mask_ln8

class Transformer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, numEncoderLayer: int, numDecoderLayer: int, dimFeedForward: int,
                 dropout: float):
        super(Transformer, self).__init__()

        encoderLayer = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder = TransformerEncoder(encoderLayer, numEncoderLayer)

        decoderLayer = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder = TransformerDecoder(decoderLayer, numDecoderLayer)

        self.resetParameters()

    def resetParameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query: Tensor, pos: Tensor) -> Tensor:
        """
        :param src: tensor of shape [batchSize, hiddenDims, imageHeight // 32, imageWidth // 32]

        :param mask: tensor of shape [batchSize, imageHeight // 32, imageWidth // 32]
                     Please refer to detr.py for more detailed description.

        :param query: object queries, tensor of shape [numQuery, hiddenDims].

        :param pos: positional encoding, the same shape as src.

        :return: tensor of shape [batchSize, numQuery * numDecoderLayer, hiddenDims]
        """
        N = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)


        pos = pos.flatten(2).permute(2, 0, 1)
        query = query.unsqueeze(1).repeat(1, N, 1)
        tgt = torch.zeros_like(query)

        memory = self.encoder(src, srcKeyPaddingMask=mask, pos=pos)
        out = self.decoder(tgt, memory, memoryKeyPaddingMask=mask, pos=pos, queryPos=query).transpose(1, 2)
        return out, memory


class TransformerEncoder(nn.Module):
    def __init__(self, encoderLayer: nn.Module, numLayers: int):
        super(TransformerEncoder, self).__init__()

        self.layers = getClones(encoderLayer, numLayers)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, srcKeyPaddingMask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        out = src

        for layer in self.layers:
            out = layer(out, mask, srcKeyPaddingMask, pos)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoderLayer: nn.Module, numLayers: int):
        super(TransformerDecoder, self).__init__()

        self.layers = getClones(decoderLayer, numLayers)

    def forward(self, tgt: Tensor, memory: Tensor, tgtMask: Optional[Tensor] = None,
                memoryMask: Optional[Tensor] = None, tgtKeyPaddingMask: Optional[Tensor] = None,
                memoryKeyPaddingMask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                queryPos: Optional[Tensor] = None) -> Tensor:
        out = tgt

        intermediate = []

        for layer in self.layers:
            out = layer(out, memory, tgtMask, memoryMask, tgtKeyPaddingMask, memoryKeyPaddingMask, pos, queryPos)
            intermediate.append(out)

        return torch.stack(intermediate)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, dimFeedForward: int, dropout: float):
        super(TransformerDecoderLayer, self).__init__()

        self.attention1 = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)

        self.linear1 = nn.Linear(hiddenDims, dimFeedForward)
        self.linear2 = nn.Linear(dimFeedForward, hiddenDims)

        self.norm1 = nn.LayerNorm(hiddenDims)
        self.norm2 = nn.LayerNorm(hiddenDims)
        self.norm3 = nn.LayerNorm(hiddenDims)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt: Tensor, memory: Tensor, tgtMask: Optional[Tensor] = None,
                memoryMask: Optional[Tensor] = None, tgtKeyPaddingMask: Optional[Tensor] = None,
                memoryKeyPaddingMask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                queryPos: Optional[Tensor] = None) -> Tensor:
        q = k = withPosEmbed(tgt, queryPos)
        tgt2 = self.attention1(q, k, value=tgt, attn_mask=tgtMask, key_padding_mask=tgtKeyPaddingMask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.attention2(query=withPosEmbed(tgt, queryPos), key=withPosEmbed(memory, pos),
                               value=memory, attn_mask=memoryMask, key_padding_mask=memoryKeyPaddingMask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt




class TransformerDecoderLayer_cross(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, dimFeedForward: int, dropout: float):
        super(TransformerDecoderLayer_cross, self).__init__()

        self.attention_cross = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)

        self.linear1 = nn.Linear(hiddenDims, dimFeedForward)
        self.linear1_1 = nn.Linear(dimFeedForward, hiddenDims)
        self.linear2 = nn.Linear(hiddenDims, dimFeedForward)
        self.linear2_1 = nn.Linear(dimFeedForward, hiddenDims)

        self.norm1 = nn.LayerNorm(hiddenDims)
        self.norm1_1 = nn.LayerNorm(hiddenDims)
        self.norm2 = nn.LayerNorm(hiddenDims)
        self.norm3 = nn.LayerNorm(hiddenDims)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt: Tensor, tgt_fuse: Tensor, memory: Tensor, tgtMask: Optional[Tensor] = None, fuseMask: Optional[Tensor] = None,
                memoryMask: Optional[Tensor] = None, tgtKeyPaddingMask: Optional[Tensor] = None,
                memoryKeyPaddingMask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                queryPos: Optional[Tensor] = None, queryPos_fuse: Optional[Tensor] = None)-> Tensor:
        tgt2 = self.attention_cross(query=withPosEmbed(tgt, queryPos), key=withPosEmbed(tgt_fuse, queryPos_fuse),
                               value=tgt_fuse, attn_mask=fuseMask)[0] #query (L): tgt64, key (S): memory:tgt128, value: tgt128
        tgt = tgt + self.dropout1(tgt2) #attn_mask (L,S)
        tgt = self.norm1(tgt)
        tgt2 = self.linear1_1(self.dropout1_1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout1_2(tgt2)
        tgt = self.norm1_1(tgt)

        tgt2 = self.attention2(query=withPosEmbed(tgt, queryPos), key=withPosEmbed(memory, pos),
                               value=memory, attn_mask=memoryMask, key_padding_mask=memoryKeyPaddingMask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2_1(self.dropout(self.activation(self.linear2(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # print("in Decoder layer tgt output:",tgt.shape)
        return tgt

class PositionEmbeddingSine(nn.Module):
    def __init__(self, numPositionFeatures: int = 64, temperature: int = 10000, normalize: bool = True,
                 scale: float = None):
        super(PositionEmbeddingSine, self).__init__()

        self.numPositionFeatures = numPositionFeatures
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        N, _, H, W = x.shape

        mask = torch.zeros(N, H, W, dtype=torch.bool, device=x.device)
        notMask = ~mask

        yEmbed = notMask.cumsum(1)
        xEmbed = notMask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            yEmbed = yEmbed / (yEmbed[:, -1:, :] + epsilon) * self.scale
            xEmbed = xEmbed / (xEmbed[:, :, -1:] + epsilon) * self.scale

        dimT = torch.arange(self.numPositionFeatures, dtype=torch.float32, device=x.device)
        dimT = self.temperature ** (2 * (dimT // 2) / self.numPositionFeatures)

        posX = xEmbed.unsqueeze(-1) / dimT
        posY = yEmbed.unsqueeze(-1) / dimT

        posX = torch.stack((posX[:, :, :, 0::2].sin(), posX[:, :, :, 1::2].cos()), -1).flatten(3)
        posY = torch.stack((posY[:, :, :, 0::2].sin(), posY[:, :, :, 1::2].cos()), -1).flatten(3)

        return torch.cat((posY, posX), 3).permute(0, 3, 1, 2), mask

class Joiner(nn.Module):
    def __init__(self, backbone: nn.Module, positionEmbedding: nn.Module):
        super(Joiner, self).__init__()

        self.backbone = backbone
        self.positionEmbedding = positionEmbedding

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        pos_embed = self.positionEmbedding(features)
        return features, pos_embed


class Joiner_conformer(nn.Module):
    def __init__(self, backbone: nn.Module, positionEmbedding: nn.Module):
        super(Joiner_conformer, self).__init__()

        self.backbone = backbone
        self.positionEmbedding = positionEmbedding

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        [features_4, features_t_4, features_8, features_t_8, features_12, features_t_12, features, features_t]  = self.backbone(x)
        pos_embed = self.positionEmbedding(features)
        return features, self.positionEmbedding(features)

class Joiner_conformer_multiproj(nn.Module):
    def __init__(self, backbone: nn.Module, positionEmbedding: nn.Module):
        super(Joiner_conformer_multiproj, self).__init__()

        self.backbone = backbone
        self.positionEmbedding = positionEmbedding

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        [features_4, features_t_4, features_8, features_t_8, features_12, features_t_12, features, features_t] = self.backbone(x)
        pos_embed = self.positionEmbedding(features)
        return features_4, features_8, features_12, features, self.positionEmbedding(features)


def buildBackbone(hiddenDims = 512):
    positionEmbedding = PositionEmbeddingSine(hiddenDims // 2)
    DLANet = dla60x(return_features = True)
    return Joiner(DLANet, positionEmbedding)

def buildBackbone_conformer_org(hiddenDims = 512):
    hiddenDims=768*2
    positionEmbedding = PositionEmbeddingSine(hiddenDims // 2)
    conformerNet = conformer(pretrained=True)
    return Joiner_conformer(conformerNet, positionEmbedding)

def buildBackbone_conformer(hiddenDims = 512, is_multi_proj=False):
    hiddenDims=768*2
    positionEmbedding = PositionEmbeddingSine(hiddenDims // 2)
    if is_multi_proj:
        conformerNet = conformer(pretrained=True,is_multi_proj=True)
        return Joiner_conformer_multiproj(conformerNet, positionEmbedding)
    else:
        conformerNet = conformer(pretrained=True)
        return Joiner_conformer(conformerNet, positionEmbedding)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, dimFeedForward: int, dropout: float):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = nn.MultiheadAttention(hiddenDims, numHead, dropout=dropout)

        self.linear1 = nn.Linear(hiddenDims, dimFeedForward)
        self.linear2 = nn.Linear(dimFeedForward, hiddenDims)

        self.norm1 = nn.LayerNorm(hiddenDims)
        self.norm2 = nn.LayerNorm(hiddenDims)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, srcKeyPaddingMask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        q = k = withPosEmbed(src, pos)
        src2 = self.attention(q, k, value=src, attn_mask=mask, key_padding_mask=srcKeyPaddingMask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def withPosEmbed(tensor: Tensor, pos: Optional[Tensor] = None) -> Tensor:
    return tensor + pos if pos is not None else tensor


def getClones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class MLP(nn.Module):
    def __init__(self, inputDim: int, hiddenDim: int, outputDim: int, numLayers: int):
        super().__init__()
        self.numLayers = numLayers

        h = [hiddenDim] * (numLayers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([inputDim] + h, h + [outputDim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.numLayers - 1 else layer(x)
        return x

def SG_upsample(dist, intensity, rgb_ratio,ln_src, ln_tgt):
    _8_to_16_neighbor_tensor = torch.tensor(_8_to_16_neighbor)
    _8_to_32_neighbor_tensor = torch.tensor(_8_to_32_neighbor)
    _16_to_32_neighbor_tensor = torch.tensor(_16_to_32_neighbor)
    _32_to_64_neighbor_tensor = torch.tensor(_32_to_64_neighbor)
    _64_to_128_neighbor_tensor = torch.tensor(_64_to_128_neighbor)
    dist_up = torch.zeros(dist.shape[0],ln_tgt)

    if ln_src==8 and ln_tgt==16:
        neighbors = _8_to_16_neighbor_tensor
    elif ln_src==8 and ln_tgt==32:
        neighbors = _8_to_32_neighbor_tensor
    elif ln_src==16 and ln_tgt==32:
        neighbors = _16_to_32_neighbor_tensor
    elif ln_src==32 and ln_tgt==64:
        neighbors = _32_to_64_neighbor_tensor
    elif ln_src==64 and ln_tgt==128:
        neighbors = _64_to_128_neighbor_tensor

    for i in range(0,dist.shape[0]):
        for j in range(0,neighbors.shape[0]):
            for k in range(0,neighbors.shape[1]):
                dist_up[i,neighbors[j,k]] = dist[i,j]/neighbors.shape[1]

    rgb_ratio_up = rgb_ratio
    intensity_up = intensity 
    return dist_up, intensity_up, rgb_ratio_up

class Transformer_hierachy_cross_conformer(nn.Module):
    def __init__(self, hiddenDims: int, numHead: int, numEncoderLayer: int, numDecoderLayer: int, dimFeedForward: int,
                 dropout: float):
        super(Transformer_hierachy_cross_conformer, self).__init__()

        self.hiddenDims = hiddenDims
        self.numHead = numHead

        encoderLayer_8 = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder_8 = TransformerEncoder(encoderLayer_8, numEncoderLayer)

        encoderLayer_16 = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder_16 = TransformerEncoder(encoderLayer_16, numEncoderLayer)

        encoderLayer_32 = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder_32 = TransformerEncoder(encoderLayer_32, numEncoderLayer)

        encoderLayer_64 = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder_64 = TransformerEncoder(encoderLayer_64, numEncoderLayer)

        encoderLayer_128 = TransformerEncoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.encoder_128 = TransformerEncoder(encoderLayer_128, numEncoderLayer)


        decoderLayer_8 = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder_8 = TransformerDecoder(decoderLayer_8, numDecoderLayer)

        self.queryEmbed_8 = nn.Embedding(9, hiddenDims)
        self.cross_atten_8 = TransformerDecoderLayer_cross(hiddenDims, numHead, dimFeedForward, dropout)


        self.fc_dist_8 = nn.Linear(hiddenDims, 1)
        self.fc_intensity_8 = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio_8 = nn.Linear(hiddenDims, 3)


        decoderLayer_16 = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder_16 = TransformerDecoder(decoderLayer_16, numDecoderLayer)
        self.cross_atten_16 = TransformerDecoderLayer_cross(hiddenDims, numHead, dimFeedForward, dropout)

        self.queryEmbed_16 = nn.Embedding(17, hiddenDims)

        self.fc_dist_16 = nn.Linear(hiddenDims, 1)
        self.fc_intensity_16 = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio_16 = nn.Linear(hiddenDims, 3)

        decoderLayer_32 = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder_32 = TransformerDecoder(decoderLayer_32, numDecoderLayer)

        self.queryEmbed_32 = nn.Embedding(33, hiddenDims)
        self.cross_atten_32 = TransformerDecoderLayer_cross(hiddenDims, numHead, dimFeedForward, dropout)


        self.fc_dist_32 = nn.Linear(hiddenDims, 1)
        self.fc_intensity_32 = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio_32 = nn.Linear(hiddenDims, 3)

        decoderLayer_64 = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder_64 = TransformerDecoder(decoderLayer_64, numDecoderLayer)

        self.queryEmbed_64 = nn.Embedding(65, hiddenDims)
        self.cross_atten_64 = TransformerDecoderLayer_cross(hiddenDims, numHead, dimFeedForward, dropout)


        self.fc_dist_64 = nn.Linear(hiddenDims, 1)
        self.fc_intensity_64 = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio_64 = nn.Linear(hiddenDims, 3)

        decoderLayer_128 = TransformerDecoderLayer(hiddenDims, numHead, dimFeedForward, dropout)
        self.decoder_128 = TransformerDecoder(decoderLayer_128, numDecoderLayer)

        self.queryEmbed_128 = nn.Embedding(129, hiddenDims)


        self.fc_dist_128 = nn.Linear(hiddenDims, 1)
        self.fc_intensity_128 = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio_128 = nn.Linear(hiddenDims, 3)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

        self.resetParameters()


        self.neighbors_16, self.neighbors_mask_16 = neighbor_mask(ln_8=8,ln_16=16,radius=0.65)
        self.neighbors_32, self.neighbors_mask_32 = neighbor_mask(ln_8=16,ln_16=32,radius=0.4)
        self.neighbors_64, self.neighbors_mask_64 = neighbor_mask(ln_8=32,ln_16=64,radius=0.3)
        self.neighbors_128, self.neighbors_mask_128 = neighbor_mask(ln_8=64,ln_16=128,radius=0.3)

    def resetParameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, pos: Tensor) -> Tensor:

        N = src.shape[0] #batch size
        neighbors_mask_16 = self.neighbors_mask_16.unsqueeze(0).repeat(N*self.numHead, 1, 1).to(src.device)
        neighbors_mask_32 = self.neighbors_mask_32.unsqueeze(0).repeat(N*self.numHead, 1, 1).to(src.device)
        neighbors_mask_64 = self.neighbors_mask_64.unsqueeze(0).repeat(N*self.numHead, 1, 1).to(src.device)
        neighbors_mask_128 = self.neighbors_mask_128.unsqueeze(0).repeat(N*self.numHead, 1, 1).to(src.device)

        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        src_128=src
        src_128 = src_128.flatten(2).permute(2, 0, 1)
        query_128 = self.queryEmbed_128.weight
        query_128 = query_128.unsqueeze(1).repeat(1, N, 1)
        tgt_128 = torch.zeros_like(query_128)

        memory_128 = self.encoder_128(src_128, srcKeyPaddingMask=mask, pos=pos)

        out_raw = self.decoder_128(tgt_128, memory_128, memoryKeyPaddingMask=mask, pos=pos, queryPos=query_128)

        out = out_raw[-1]
        out = out.transpose(0, 1)
        out_1 = out[:,1:,:] 
        out_2 = out[:,:1,:]


        dist_pred = self.fc_dist_128(out_1).squeeze()
        if len(dist_pred.shape)<2:
            dist_pred = dist_pred.unsqueeze(0)
        dist_pred = self.sigmoid(dist_pred)

        intenstiy_pred = self.fc_intensity_128(out_2)

        rgb_ratio_pred = self.fc_rgb_ratio_128(out_2)
        rgb_ratio_pred = self.sigmoid(rgb_ratio_pred)
        pred = {'distribution': dist_pred,
                    'intensity': intenstiy_pred,
                    'rgb_ratio': rgb_ratio_pred
                    }

        src_64=src
        src_64 = src_64.flatten(2).permute(2, 0, 1)
        query_64 = self.queryEmbed_64.weight
        query_64 = query_64.unsqueeze(1).repeat(1, N, 1)
        tgt_64 = torch.zeros_like(query_64)

        memory_64 = self.encoder_64(src_64, srcKeyPaddingMask=mask, pos=pos)

        out_64_raw = self.decoder_64(tgt_64, memory_64, memoryKeyPaddingMask=mask, pos=pos, queryPos=query_64)
        out_64_cross = self.cross_atten_64(out_64_raw[-1], out_raw[-1], memory_64, memoryKeyPaddingMask=mask, fuseMask=neighbors_mask_128, queryPos=query_64, pos=pos, queryPos_fuse=query_128 ) 
        out_64 = out_64_cross

        if len(out_64.shape)<3:
            out_64 = out_64.unsqueeze(1)   

        out_64 = out_64.transpose(0, 1)
        out_64_1 = out_64[:,1:,:]

        out_64_temp = out_64_raw[-1].transpose(0, 1)
        out_64_2 = out_64_temp[:,:1,:]

        dist_pred_64 = self.fc_dist_64(out_64_1).squeeze()
        if len(dist_pred_64.shape)<2:
            dist_pred_64 = dist_pred_64.unsqueeze(0)

        dist_pred_64 = self.sigmoid(dist_pred_64)
        intenstiy_pred_64 = self.fc_intensity_64(out_64_2)

        rgb_ratio_pred_64 = self.fc_rgb_ratio_64(out_64_2)
        rgb_ratio_pred_64 = self.sigmoid(rgb_ratio_pred_64)

        pred_64 = {'distribution': dist_pred_64,
                    'intensity': intenstiy_pred_64,
                    'rgb_ratio': rgb_ratio_pred_64
                    }

        src_32=src
        src_32 = src_32.flatten(2).permute(2, 0, 1)
        query_32 = self.queryEmbed_32.weight
        query_32 = query_32.unsqueeze(1).repeat(1, N, 1)
        tgt_32 = torch.zeros_like(query_32)

        memory_32 = self.encoder_32(src_32, srcKeyPaddingMask=mask, pos=pos)
        out_32_raw = self.decoder_32(tgt_32, memory_32, memoryKeyPaddingMask=mask, pos=pos, queryPos=query_32)

        out_32_cross = self.cross_atten_32(out_32_raw[-1], out_64_raw[-1], memory_32, memoryKeyPaddingMask=mask, fuseMask=neighbors_mask_64, queryPos=query_32, pos=pos, queryPos_fuse =query_64 )
        out_32 = out_32_cross

        if len(out_32.shape)<3:
            out_32 = out_32.unsqueeze(1)  

        out_32 = out_32.transpose(0, 1)
        out_32_1 = out_32[:,1:,:]

        out_32_temp = out_32_raw[-1].transpose(0, 1)
        out_32_2 = out_32_temp[:,:1,:]

        dist_pred_32 = self.fc_dist_32(out_32_1).squeeze()
        if len(dist_pred_32.shape)<2:
            dist_pred_32 = dist_pred_32.unsqueeze(0)

        dist_pred_32 = self.sigmoid(dist_pred_32)
        intenstiy_pred_32 = self.fc_intensity_32(out_32_2)

        rgb_ratio_pred_32 = self.fc_rgb_ratio_32(out_32_2)
        rgb_ratio_pred_32 = self.sigmoid(rgb_ratio_pred_32)

        pred_32 = {'distribution': dist_pred_32,
                    'intensity': intenstiy_pred_32,
                    'rgb_ratio': rgb_ratio_pred_32
                    }

        src_16=src
        src_16 = src_16.flatten(2).permute(2, 0, 1)
        query_16 = self.queryEmbed_16.weight
        query_16 = query_16.unsqueeze(1).repeat(1, N, 1)
        tgt_16 = torch.zeros_like(query_16)#[9, 32, 512]  
        memory_16 = self.encoder_16(src_16, srcKeyPaddingMask=mask, pos=pos)
        out_16_raw = self.decoder_16(tgt_16, memory_16, memoryKeyPaddingMask=mask, pos=pos, queryPos=query_16)
        out_16_cross = self.cross_atten_16(out_16_raw[-1], out_32_raw[-1], memory_16, memoryKeyPaddingMask=mask, fuseMask=neighbors_mask_32, queryPos=query_16, pos=pos, queryPos_fuse =query_32 )
        out_16 = out_16_cross

        if len(out_16.shape)<3:
            out_16 = out_16.unsqueeze(1)  
        out_16 = out_16.transpose(0, 1)
        out_16_1 = out_16[:,1:,:]

        out_16_temp = out_16_raw[-1].transpose(0, 1)
        out_16_2 = out_16_temp[:,:1,:]

        dist_pred_16 = self.fc_dist_16(out_16_1).squeeze()
        if len(dist_pred_16.shape)<2:
            dist_pred_16 = dist_pred_16.unsqueeze(0)

        dist_pred_16 = self.sigmoid(dist_pred_16)
        intenstiy_pred_16 = self.fc_intensity_16(out_16_2)
        rgb_ratio_pred_16 = self.fc_rgb_ratio_8(out_16_2)
        rgb_ratio_pred_16 = self.sigmoid(rgb_ratio_pred_16)

        pred_16 = {'distribution': dist_pred_16,
                    'intensity': intenstiy_pred_16,
                    'rgb_ratio': rgb_ratio_pred_16
                    }

        src_8=src
        src_8 = src_8.flatten(2).permute(2, 0, 1)
        query_8 = self.queryEmbed_8.weight
        query_8 = query_8.unsqueeze(1).repeat(1, N, 1)
        tgt_8 = torch.zeros_like(query_8)#[9, 32, 512]
        memory_8 = self.encoder_8(src_8, srcKeyPaddingMask=mask, pos=pos)
        out_8_raw = self.decoder_8(tgt_8, memory_8, memoryKeyPaddingMask=mask, pos=pos, queryPos=query_8)
        out_8_cross = self.cross_atten_8(out_8_raw[-1], out_16_raw[-1], memory_8, memoryKeyPaddingMask=mask, fuseMask=neighbors_mask_16, queryPos=query_8, pos=pos, queryPos_fuse =query_16 )
        out_8 = out_8_raw
        if len(out_8.shape)<3:
            out_8 = out_8.unsqueeze(1)
        out_8 = out_8.transpose(1, 2)
        out_8_1 = out_8[:,1:,:]
        out_8_temp = out_8_raw[-1].transpose(0, 1)
        out_8_2 = out_8_temp[:,:1,:]

        dist_pred_8 = self.fc_dist_8(out_8_1).squeeze()
        if len(dist_pred_8.shape)<2:
            dist_pred_8 = dist_pred_8.unsqueeze(0)
        dist_pred_8 = self.sigmoid(dist_pred_8)
        intenstiy_pred_8 = self.fc_intensity_8(out_8_2)
        rgb_ratio_pred_8 = self.fc_rgb_ratio_8(out_8_2)
        rgb_ratio_pred_8 = self.sigmoid(rgb_ratio_pred_8)

        pred_8 = {'distribution': dist_pred_8,
                    'intensity': intenstiy_pred_8,
                    'rgb_ratio': rgb_ratio_pred_8
                    }

        return  pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out

class DETR_hierachy(nn.Module):
    def __init__(self, hiddenDims = 512,numQuery=129,numClass=5):
        super(DETR_hierachy, self).__init__()

        self.backbone = buildBackbone()

        self.reshape = nn.Conv2d(1024, hiddenDims, 1)

        self.transformer = Transformer_hierachy(hiddenDims, numHead=8, numEncoderLayer=6, numDecoderLayer=6,
                                       dimFeedForward=2048, dropout=0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        features, (pos, mask) = self.backbone(x)
        features = self.reshape(features)

        pred_8, pred_32, pred_64, pred, out_8, out_32, out_64, out, memory= self.transformer(features, mask, pos)
       

        return pred_8, pred_32, pred_64, pred, out_8, out_32, out_64, out, memory,features



class Salenet_SG(nn.Module):
    def __init__(self, hiddenDims = 512,numQuery=129,numClass=5):
        super(Salenet_SG, self).__init__()

        hiddenDims=768*2
        self.backbone = buildBackbone_conformer()

        self.transformer = Transformer_hierachy_cross_conformer(hiddenDims, numHead=8, numEncoderLayer=6, numDecoderLayer=6,
                                       dimFeedForward=2048, dropout=0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:

        features, (pos, mask) = self.backbone(x)

        pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out= self.transformer(features, mask, pos)

        return pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out,features

class DETR_hierachy_cross_attention_conformer(nn.Module):
    def __init__(self, hiddenDims = 512,numQuery=129,numClass=5):
        super(DETR_hierachy_cross_attention_conformer, self).__init__()
        hiddenDims=768*2
        self.backbone = buildBackbone_conformer()
        self.transformer = Transformer_hierachy_cross_conformer(hiddenDims, numHead=8, numEncoderLayer=6, numDecoderLayer=6,
                                       dimFeedForward=2048, dropout=0.1)

        self.projhead_avgpool = nn.AvgPool2d(8)
        self.projhead_fc1 =  nn.Linear(1536, 1024, bias = False)
        self.projhead_relu =  nn.LeakyReLU(inplace=True)
        self.projhead_fc2 = nn.Linear(1024, 512, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:

        features, (pos, mask) = self.backbone(x)
        projfeatures = self.projhead_avgpool(features).squeeze()
        projfeatures = self.projhead_fc1(projfeatures)
        projfeatures = self.projhead_relu(projfeatures)
        projfeatures = self.projhead_fc2(projfeatures)

        pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out= self.transformer(features, mask, pos)
       

        return pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out,features, projfeatures


class DETR_hierachy_cross_attention(nn.Module):
    def __init__(self, hiddenDims = 512,numQuery=129,numClass=5):
        super(DETR_hierachy_cross_attention, self).__init__()

        self.backbone = buildBackbone()

        self.transformer = Transformer_hierachy_cross(hiddenDims, numHead=8, numEncoderLayer=6, numDecoderLayer=6,
                                       dimFeedForward=2048, dropout=0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:

        features, (pos, mask) = self.backbone(x)
        print("IN DETR_hierachy_cross_attention feature shape after reshape:",features.shape)

        pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out= self.transformer(features, mask, pos)
       
        return pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out,features

class DETR(nn.Module):
    def __init__(self, hiddenDims = 512,numQuery=129,numClass=5,useDepth = True):
        super(DETR, self).__init__()

        self.backbone = buildBackbone()

        self.reshape = nn.Conv2d(1024, hiddenDims, 1)

        self.transformer = Transformer(hiddenDims, numHead=8, numEncoderLayer=6, numDecoderLayer=6,
                                       dimFeedForward=2048, dropout=0.1)

        self.queryEmbed = nn.Embedding(numQuery, hiddenDims)
        self.fc_dist = nn.Linear(hiddenDims, 1)
        self.fc_intensity = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio = nn.Linear(hiddenDims, 3)
        self.fc_ambient = nn.Linear(hiddenDims, 3)
        self.fc_depth = nn.Linear(hiddenDims, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.useDepth = useDepth

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        features, (pos, mask) = self.backbone(x)
        features = self.reshape(features)

        out,memory = self.transformer(features, mask, self.queryEmbed.weight, pos)
        out = out[-1].squeeze()

        print("out shape:",out.shape)
        if out.dim()<3:
            out = out.unsqueeze(0)

        out_1 = out[:,1:,:]
        out_2 = out[:,:1,:].squeeze()
        dist_pred = self.fc_dist(out_1).squeeze()
        dist_pred = self.sigmoid(dist_pred)
        if dist_pred.dim()<2:
            dist_pred = dist_pred.unsqueeze(0)
        intenstiy_pred = self.fc_intensity(out_2)
        print("intensity_pred shape:",intenstiy_pred.shape)
        if intenstiy_pred.dim()<2:
            intenstiy_pred = intenstiy_pred.unsqueeze(0)

        rgb_ratio_pred = self.fc_rgb_ratio(out_2)
        rgb_ratio_pred = self.sigmoid(rgb_ratio_pred)
        print("rgb_ratio_pred shape:",rgb_ratio_pred.shape)

        if rgb_ratio_pred.dim()<2:
            rgb_ratio_pred = rgb_ratio_pred.unsqueeze(0)


        ambient_pred = self.fc_ambient(out_2)
        ambient_pred = self.relu(ambient_pred)
        print("ambient_pred shape:",ambient_pred.shape)
        if ambient_pred.dim()<2:
            ambient_pred = ambient_pred.unsqueeze(0)

        print("dist_pred shape:",dist_pred.shape)
        if self.useDepth:
            depth_pred = self.fc_depth(out_1).squeeze()
            #depth_pred = self.sigmoid(depth_pred)
            
            return {'distribution': dist_pred,
                    'intensity': intenstiy_pred,
                    'rgb_ratio': rgb_ratio_pred,
                    'ambient': ambient_pred,
                    'depth': depth_pred
                    }
        else:
            return {'distribution': dist_pred,
                    'intensity': intenstiy_pred,
                    'rgb_ratio': rgb_ratio_pred,
                    'ambient': ambient_pred
                    },features,memory

class DETR_conformer(nn.Module):
    def __init__(self, hiddenDims = 512,numQuery=129,numClass=5,useDepth = True):
        super(DETR_conformer, self).__init__()
        hiddenDims=768*2

        self.backbone = buildBackbone_conformer()
        self.transformer = Transformer_onlyDecoder(hiddenDims, numHead=8, numDecoderLayer=6,
                                       dimFeedForward=2048, dropout=0.1)

        self.queryEmbed = nn.Embedding(numQuery, hiddenDims)
        self.fc_dist = nn.Linear(hiddenDims, 1)
        self.fc_intensity = nn.Linear(hiddenDims, 1)
        self.fc_rgb_ratio = nn.Linear(hiddenDims, 3)
        self.fc_ambient = nn.Linear(hiddenDims, 3)
        self.fc_depth = nn.Linear(hiddenDims, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.useDepth = useDepth

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:

        features, (pos, mask) = self.backbone(x)

        out,memory = self.transformer(features, mask, self.queryEmbed.weight, pos)

        out = out[-1].squeeze()
        if out.dim()<3:
            out = out.unsqueeze(0)


        out_1 = out[:,1:,:]
        out_2 = out[:,:1,:].squeeze()

        dist_pred = self.fc_dist(out_1).squeeze()
        dist_pred = self.sigmoid(dist_pred)
        if dist_pred.dim()<2:
            dist_pred = dist_pred.unsqueeze(0)

        intenstiy_pred = self.fc_intensity(out_2)
        if intenstiy_pred.dim()<2:
            intenstiy_pred = intenstiy_pred.unsqueeze(0)
        rgb_ratio_pred = self.fc_rgb_ratio(out_2)
        rgb_ratio_pred = self.sigmoid(rgb_ratio_pred)
        if rgb_ratio_pred.dim()<2:
            rgb_ratio_pred = rgb_ratio_pred.unsqueeze(0)
        ambient_pred = self.fc_ambient(out_2)
        ambient_pred = self.relu(ambient_pred)
        if ambient_pred.dim()<2:
            ambient_pred = ambient_pred.unsqueeze(0)
        if self.useDepth:
            depth_pred = self.fc_depth(out_1).squeeze()
            #depth_pred = self.sigmoid(depth_pred)
            
            return {'distribution': dist_pred,
                    'intensity': intenstiy_pred,
                    'rgb_ratio': rgb_ratio_pred,
                    'ambient': ambient_pred,
                    'depth': depth_pred
                    }
        else:
            return {'distribution': dist_pred,
                    'intensity': intenstiy_pred,
                    'rgb_ratio': rgb_ratio_pred,
                    'ambient': ambient_pred
                    },features,memory

