import torch
import torch.nn as nn

def BCE_loss(pred, label):
    bce_loss = nn.BCELoss()
    return bce_loss(pred, label)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

class LossG(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.DiceLoss = DiceLoss()

    def forward(self, outputs, inputs):
        losses = {}
        loss_G = 0
        preds = outputs['preds']
        inputs = (inputs > 0).float()
        

        if self.cfg['loss'] == 'BCE':
            losses['loss_BCE'] = BCE_loss(preds, inputs)
            loss_G += losses['loss_BCE']
        elif self.cfg['loss'] == 'Dice':
            losses['loss_Dice'] = self.DiceLoss(inputs, preds)
            loss_G += losses['loss_Dice']

        losses['loss'] = loss_G
        return losses