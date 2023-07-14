import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=1,act=None ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3,stride=1, padding = 1, bias=False)
        self.norm = torch.nn.InstanceNorm3d(out_ch)
        self.act  = nn.SiLU()
        self.conv2 = nn.Conv3d(in_channels = out_ch, out_channels = out_ch, kernel_size = k,stride=s, padding = 1, bias=False)
        self.norm2 = torch.nn.InstanceNorm3d(out_ch)
    def forward(self, x):
        x =  self.act(self.norm(self.conv(x)))
        x =  self.act(self.norm2(self.conv2(x)))
        return x
    
    
class ConvBlock2(nn.Module):
    def __init__(self,in_ch,out_ch,act=None ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, padding = 1, bias=False)
        self.norm = torch.nn.InstanceNorm3d(out_ch)
        self.act   = nn.SiLU() 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class DecoderBlock(nn.Module):
    def __init__(
        self,in_ch,skip_ch,out_ch,k=(2,1,1),s=(2,1,1),p=(1,0,0)):
        super().__init__()
    
        self.conv1 = ConvBlock2(in_ch + skip_ch,out_ch,act = True)
        self.trans = torch.nn.ConvTranspose3d(in_ch, in_ch, kernel_size=k, stride=s,padding=p )
    def forward(self, x, skip=None):
        x =  self.trans(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, padding = 1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class DecoderBlock2d(nn.Module):
    def __init__(
        self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock2d(in_ch + skip_ch,out_ch)
        self.conv2 = ConvBlock2d(out_ch,out_ch,)
        self.trans = torch.nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2, padding=0)
    def forward(self, x, skip=None):
        x = self.trans(x)         
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Model2d(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        url = 'https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/releases/download/full_effb0_ns/Full_Efficient_B0_3ch.pt'
        model_file = 'fweights.pt'
        torch.hub.download_url_to_file(url, model_file)
        self.encoder = torch.load(model_file)
        self.decoder1 = DecoderBlock2d(320,112,256)
        self.decoder2 = DecoderBlock2d(256,40,128)
        self.decoder3 = DecoderBlock2d(128,24,64)
        self.decoder4 = DecoderBlock2d(64,32,32)
        self.decoder5 = DecoderBlock2d(32,0,16)
           
        self.last = nn.Conv2d(in_channels = 16, out_channels = out_ch, kernel_size = 3, padding = 1)
   
    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder1(h[4],h[3])
        y = self.decoder2(y,h[2])
        y = self.decoder3(y,h[1])
        y = self.decoder4(y,h[0])
        y = self.decoder5(y)
        
        y = self.last(y)
        return y

class Model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=in_ch,out_channels=16,kernel_size=3,stride=1, padding=1,dilation=1,bias=False)
        self.norm1 = torch.nn.InstanceNorm3d(16, eps=1e-05, momentum=0.1)
        self.act1 = nn.SiLU(inplace=True) 
        self.conv2 = torch.nn.Conv3d(in_channels=16,out_channels=32,kernel_size=3,stride=(3,1,1), padding=1,dilation=1,bias=False)
        self.norm2 = torch.nn.InstanceNorm3d(32, eps=1e-05, momentum=0.1)
        self.act2 = nn.SiLU(inplace=True)
        
        self.enc3 = nn.Sequential(
         ConvBlock(32,48,s=(2,1,1)),
        )
        self.enc4 = nn.Sequential(
         ConvBlock(48,96,s=(2,1,1)),
        )
        self.enc5 = nn.Sequential(
          ConvBlock(in_ch=96,out_ch=4,s=(4,1,1)),
        )
        
        self.model2D = Model2d(out_ch = 3).to(device)
     
        self.dec1 = DecoderBlock(4,96,96,k=(5,1,1),s=(3,1,1),p=(0,0,0))
        self.dec2 = DecoderBlock(96,48,48,k=(2,1,1),s=(2,1,1),p=(0,0,0))
        self.dec3 = DecoderBlock(48,32,32,k=(3,1,1),s=(2,1,1),p=(1,0,0))
        self.dec4 = DecoderBlock(32,16,16,k=(4,1,1),s=(3,1,1),p=(1,0,0))
        
        self.last = nn.Conv3d(in_channels = 16, out_channels = out_ch, kernel_size = 3, padding = 1)

    def forward(self, x):
        x1 = self.act1(self.norm1(self.conv1(x)))
        x2 = self.act2(self.norm2(self.conv2(x1)))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.enc5(x4)

        x = torch.squeeze(x)
        x = self.model2D(x)
        x = torch.unsqueeze(x,0)

        y1 = self.dec1(x,x4)
        y2 = self.dec2(y1,x3)
        y3 = self.dec3(y2,x2)
        y4 = self.dec4(y3,x1)

        return self.last(y4)

model = Model(in_ch = 4, out_ch = 3).to(device)

