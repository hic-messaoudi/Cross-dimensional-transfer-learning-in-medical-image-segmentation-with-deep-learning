import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model) :
    url = 'https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/releases/download/effb0_3ch/Efficient_B0_3ch.pt'
    weights_file = 'weights.pt'
    torch.hub.download_url_to_file(url, weights_file)
    weights = torch.load(weights_file)

    if model.in_ch == 1 : 
        weights['conv1.weight'] = weights['conv1.weight'].sum(1, keepdim=True)
    elif model.in_ch == 3 : 
        pass
    else : 
        r = weights['conv1.weight'][:,0:1] 
        weights['conv1.weight'] = torch.randn((r.shape[0], model.in_ch, r.shape[2], r.shape[3])).to(device) 
    model.load_state_dict(weights)
    return model


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """ https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """ https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class SE2D(nn.Module):
    """ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_blocks.py """
    def __init__(self, in_ch, se_ratio=0.25, act_layer=nn.SiLU, gate_fn=nn.Sigmoid(), divisor=1):
        super(SE2D, self).__init__()
        reduced_ch = int(in_ch * se_ratio / divisor)
        self.conv_reduce = nn.Conv2d(in_ch, reduced_ch, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_ch, in_ch, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)

class DepthwiseSeparableConv2D(nn.Module):
    """ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_blocks.py """
    def __init__(self, in_ch, out_ch, dw_kernel_size=3, stride=1, dilation=1, group_size=1, noskip=False, 
                 pw_kernel_size=1, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d, se_layer=SE2D, dprate=0.):
        super(DepthwiseSeparableConv2D, self).__init__()
       
        self.has_skip = (stride == 1 and in_ch == out_ch) and not noskip
        self.padding = dw_kernel_size//2
        self.conv_dw = torch.nn.Conv2d( in_ch, in_ch, dw_kernel_size, stride=stride, dilation=dilation, padding=self.padding, groups=in_ch,bias=False)
        self.bn1 = norm_layer(in_ch)
        self.act = act_layer(True)

        # Squeeze-and-excitation
        self.se = se_layer(in_ch, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = torch.nn.Conv2d(in_ch, out_ch, pw_kernel_size, padding=0,bias=False)
        self.bn2 = norm_layer(out_ch)
        self.drop_path = DropPath(dprate) if dprate and self.training else nn.Identity()


    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual2D(nn.Module):
    """ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_blocks.py """
    def __init__(
            self, in_ch, out_ch, dw_kernel_size=3, stride=1, dilation=1, group_size=1,noskip=False,
            exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=1/24, act_layer=nn.SiLU, 
            norm_layer=torch.nn.BatchNorm2d, se_layer=SE2D,dprate=None):
        super(InvertedResidual2D, self).__init__()
     
        mid_ch = in_ch * exp_ratio
      
        self.has_skip = (in_ch == out_ch and stride == 1) and not noskip
        self.padding = exp_kernel_size//2
        self.padding2 = dw_kernel_size // 2
        # Point-wise expansion
        self.conv_pw = torch.nn.Conv2d(in_channels = in_ch, out_channels = mid_ch, 
                                       kernel_size=exp_kernel_size, padding=self.padding,bias=False)
        self.bn1 = norm_layer(mid_ch)

        self.act1 = act_layer(True)
        # Depth-wise convolution
        self.conv_dw = torch.nn.Conv2d(
            mid_ch, mid_ch, dw_kernel_size, stride=stride, dilation=dilation,
            groups=mid_ch, padding=self.padding2,bias=False)
        self.bn2 = norm_layer(mid_ch)
        self.act2 = act_layer(True)

        # Squeeze-and-excitation
        self.se = se_layer(mid_ch, act_layer=act_layer,se_ratio=se_ratio) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = torch.nn.Conv2d(mid_ch, out_ch, pw_kernel_size, padding=0,bias=False)
        self.bn3 = norm_layer(out_ch)
        self.drop_path = DropPath(dprate) if dprate and self.training else nn.Identity()


    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EfficientNet2D(nn.Module):
  def __init__(self,in_ch, drop_path_rate=0.2):
    super(EfficientNet2D,self).__init__()
    self.in_ch = in_ch
    self.conv1 = torch.nn.Conv2d(in_channels=in_ch,out_channels=32,kernel_size=3,stride=2, padding=1,dilation=1,bias=False)
    self.bnorm1 = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.act1 = nn.SiLU(True)

    self.enc0 = nn.Sequential(
          DepthwiseSeparableConv2D(in_ch = 32, out_ch = 16, stride=1)
        )
    self.enc1 = nn.Sequential(
          InvertedResidual2D(in_ch=16,out_ch=24,exp_ratio=6,stride=2,se_ratio=1/24, dprate=drop_path_rate * 1 / 16),
          InvertedResidual2D(in_ch=24,out_ch=24,exp_ratio=6,stride=1,se_ratio=1/24, dprate=drop_path_rate * 2 / 16)
        )
    self.enc2 = nn.Sequential(
          InvertedResidual2D(in_ch=24,out_ch=40,exp_ratio=6,stride=2,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 3 / 16),
          InvertedResidual2D(in_ch=40,out_ch=40,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 4 / 16)
        )
    self.enc3 = nn.Sequential(
          InvertedResidual2D(in_ch=40,out_ch=80,exp_ratio=6,stride=2,se_ratio=1/24, dprate=drop_path_rate * 5 / 16),
          InvertedResidual2D(in_ch=80,out_ch=80,exp_ratio=6,stride=1,se_ratio=1/24, dprate=drop_path_rate * 6 / 16),
          InvertedResidual2D(in_ch=80,out_ch=80,exp_ratio=6,stride=1,se_ratio=1/24, dprate=drop_path_rate * 7 / 16)
        )
    self.enc4 = nn.Sequential(
          InvertedResidual2D(in_ch=80,out_ch=112,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 8 / 16),
          InvertedResidual2D(in_ch=112,out_ch=112,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 9 / 16),
          InvertedResidual2D(in_ch=112,out_ch=112,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 10 / 16)
        )
    self.enc5 = nn.Sequential(
          InvertedResidual2D(in_ch=112,out_ch=192,exp_ratio=6,stride=2,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 11 / 16),
          InvertedResidual2D(in_ch=192,out_ch=192,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 12 / 16),
          InvertedResidual2D(in_ch=192,out_ch=192,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 13 / 16),
          InvertedResidual2D(in_ch=192,out_ch=192,exp_ratio=6,stride=1,dw_kernel_size=5,se_ratio=1/24, dprate=drop_path_rate * 14 / 16)
        )
    self.enc6 = nn.Sequential(
          InvertedResidual2D(in_ch=192,out_ch=320,exp_ratio=6,stride=1,se_ratio=1/24, dprate=drop_path_rate * 15 / 16)
        )
    self.conv_head = nn.Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.norm_head = nn.BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.act_head = nn.SiLU(inplace=True)
    self.globalpool = nn.AdaptiveAvgPool2d(1280)
  
  def forward(self,x):
    x1 = self.act1(self.bnorm1(self.conv1(x)))
    x = self.enc0(x1)
    x2 = self.enc1(x)
    x3 = self.enc2(x2)
    x = self.enc3(x3)
    x4 = self.enc4(x)
    x = self.enc5(x4)
    x6 = self.enc6(x)
    return x1,x2,x3,x4,x6

Efficient = EfficientNet2D(in_ch= 3, drop_path_rate  = 0.4)

# For pre-trained encoder : 

#Efficient = load_model(EfficientNet2D(in_ch= 1, drop_path_rate  = 0.4)


