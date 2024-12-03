import torch
from models.UNet_3Plus.UNet_3Plus import UNet_3Plus
from models.img_seg_models import U_Net, AttU_Net, R2AttU_Net, R2U_Net

class Model(torch.nn.Module):
    def __init__(self, model_type, in_channels, n_classes):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'UNet':
            self.netG = U_Net(img_ch=in_channels, output_ch=n_classes).to(device)
        elif model_type == 'R2UNet':
            self.netG = R2U_Net(img_ch=in_channels, output_ch=n_classes).to(device)
        elif model_type == 'AttUNet':
            self.netG = AttU_Net(img_ch=in_channels, output_ch=n_classes).to(device)
        elif model_type == 'R2AttUNet':
            self.netG = R2AttU_Net(img_ch=in_channels, output_ch=n_classes, t=2).to(device)
        elif model_type == 'UNet3+':
            self.netG = UNet_3Plus(in_channels=in_channels, n_classes=n_classes, filters=[64,128,256]).to(device)
            
    def forward(self, input):
        outputs = {}
        outputs['preds'] = torch.sigmoid(self.netG(input))
        return outputs
