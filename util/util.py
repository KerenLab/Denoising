from torch.optim import lr_scheduler
import torch
import os
import pandas as pd

def get_scheduler(optimizer, cfg):
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    return scheduler


def get_optimizer(cfg, params):
    optimizer = torch.optim.Adam(params, lr=cfg['lr'])
    return optimizer


def save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='latest'):
    # Create a dictionary containing the current state of the model, optimizer, and other relevant values
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": current_epoch,
        "optimal_validation_loss": optimal_validation_loss,
    }
    
    # Create the path for the checkpoint file using the provided logging directory and save_name
    checkpoint_path = os.path.join(outputs_dir, "{}-{}.pt".format(model_name, save_name))
    
    # Save the checkpoint dictionary to the specified file path
    torch.save(checkpoint_state, checkpoint_path)
