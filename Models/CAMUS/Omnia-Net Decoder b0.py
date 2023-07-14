class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, padding = 1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class DecoderBlock(nn.Module):
    def __init__(
        self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch + skip_ch,out_ch)
        self.conv2 = ConvBlock(out_ch,out_ch,)
       # self.trans = torch.nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2, padding=0)
    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        # self.trans(x)         
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,16,3,stride=1,padding=1, bias=False)
        self.norm = torch.nn.BatchNorm2d(16)
        self.act = nn.ReLU()

        self.encoder = load_model(EfficientNet2D(in_ch= 16, drop_path_rate  = 0.4))
        self.decoder1 = DecoderBlock(320,112,256)
        self.decoder2 = DecoderBlock(256,40,128)
        self.decoder3 = DecoderBlock(128,24,64)
        self.decoder4 = DecoderBlock(64,32,32)
        self.decoder5 = DecoderBlock(32,16,16)
           
        self.last = nn.Conv2d(in_channels = 16, out_channels = out_ch, kernel_size = 3, padding = 1)
   
    def forward(self, x):
        n = self.act(self.norm(self.conv(x)))
        h = self.encoder(n)
        y = self.decoder1(h[4],h[3])
        y = self.decoder2(y,h[2])
        y = self.decoder3(y,h[1])
        y = self.decoder4(y,h[0])
        y = self.decoder5(y,n)
        
        y = self.last(y)
        return y
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(in_ch = 1, out_ch = 5).to(device)
