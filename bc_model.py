from turtle import back
import torch
import torch.nn as nn
from torchvision import models as vision_models
import torch.nn.functional as F
import numpy as np
import math

def normalize(data: torch.Tensor, min: torch.Tensor, max: torch.Tensor) -> torch.Tensor:
    return ((2.0 * (data - min) / (max - min)) - 1.0)

def unnormalize(data_norm: torch.Tensor, min: torch.Tensor, max: torch.Tensor) -> torch.Tensor:
    return ((0.5 * (data_norm + 1.0) * (max - min)) + min)


class CoordConv2d(nn.Conv2d, nn.Module):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)

    Implementation taken from https://github.com/ARISE-Initiative/robomimic
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert(coord_encoding in ['position'])
        self.coord_encoding = coord_encoding
        if coord_encoding == 'position':
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception("CoordConv2d: coord encoding {} not implemented".format(self.coord_encoding))
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf

    Implementation taken from https://github.com/ARISE-Initiative/robomimic
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class ResNet18Conv(nn.Module):
    """
    A ResNet18 block that can be used to process input images.

    Implementation taken from https://github.com/ARISE-Initiative/robomimic
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(weights=(vision_models.ResNet18_Weights.DEFAULT if pretrained else None))

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer and avg pool
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)

    def forward(self, inputs):
        return self.nets(inputs)


class BCVisionModel(nn.Module):
    """
    Minimal behavior cloning model:
      inputs: image (3xHxW), state (pos[2]), command [object_color, goal_color] (2)
      output: delta_pos[2] between current and next timestep
      vision encoder: ResNet18
    """

    def __init__(self):
        super().__init__()
        img_input_shape = [3,128,128]
        img_encoded_dim = 64
        self.hidden_dim = 256

        self.nets = nn.ModuleDict()

        img_encoder_net_list = []
        img_encoder_net_list.append(ResNet18Conv())
        feat_shape = img_encoder_net_list[-1].output_shape(img_input_shape)
        img_encoder_net_list.append(SpatialSoftmax(feat_shape))
        feat_shape = img_encoder_net_list[-1].output_shape(feat_shape)
        img_encoder_net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))
        img_encoder_net_list.append(torch.nn.Linear(int(np.prod(feat_shape)), img_encoded_dim))
        self.nets['img_encoder'] = nn.Sequential(*img_encoder_net_list)

        action_head_net_list = []
        action_head_net_list.extend([nn.Linear(img_encoded_dim + 4, self.hidden_dim), nn.ReLU(inplace=True)])
        for _ in range(2):
            action_head_net_list.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True)])
        action_head_net_list.append(nn.Linear(self.hidden_dim, 2))
        self.nets['action_head'] = nn.Sequential(*action_head_net_list)

        # normalization buffers
        self.register_buffer('state_min', torch.zeros(2))
        self.register_buffer('state_max', torch.ones(2))
        self.register_buffer('dpos_min', torch.zeros(2))
        self.register_buffer('dpos_max', torch.ones(2))

    def forward(self, img: torch.Tensor, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        z = self.nets['img_encoder'](img)
        x = torch.cat([z, state, command], dim=-1)
        pred = self.nets['action_head'](x)
        return pred

class BCStateModel(nn.Module):
    """
    Minimal behavior cloning model:
      inputs: image (3xHxW), state (pos[2]), command [object_color, goal_color] (2)
      output: delta_pos[2] between current and next timestep
    """

    def __init__(self):
        super().__init__()
        self.obs_dim = 2 + 2 + 2 + 2 # agent | object | goal | command
        self.hidden_dim = 512

        self.nets = nn.ModuleDict()
        action_head_net_list = []
        action_head_net_list.extend([nn.Linear(self.obs_dim, self.hidden_dim), nn.ReLU(inplace=True)])
        for _ in range(2):
            action_head_net_list.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True)])
        action_head_net_list.append(nn.Linear(self.hidden_dim, 2))
        self.nets['action_head'] = nn.Sequential(*action_head_net_list)

        # normalization buffers
        self.register_buffer('state_min', torch.zeros(2))
        self.register_buffer('state_max', torch.ones(2))
        self.register_buffer('dpos_min', torch.zeros(2))
        self.register_buffer('dpos_max', torch.ones(2))

    def forward(self, obs) -> torch.Tensor:
        pred = self.nets['action_head'](obs)
        return pred

BCModel = BCVisionModel
