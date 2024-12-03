import os
import math
import time
import torch
import numpy as np
import random
import yaml
from torch.utils.data import DataLoader
from Dataset import Denoising_Dataset
import wandb
from models.model import Model
from util.losses import LossG
from util.util import *

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(callback=None):
    # load config file
    with open("config/config_train.yaml", "r") as f:
        config = yaml.safe_load(f)
    epsilon = 1e-7

    # wandb initializations
    model_name = config['model_name']
    if model_name == 'delete':
        wandb.init(project='Denoising',name=model_name, entity=cfg['W&B_entity'], config=config)
    else:    
        wandb.init(project='Denoising',name=model_name, id=model_name, entity=cfg['W&B_entity'], config=config, resume="allow")
    cfg = wandb.config
    outputs_dir = os.path.join('pretrained_models', wandb.run.name)
    os.makedirs(outputs_dir, exist_ok=True)

    # set seed
    seed = np.random.randint(2 ** 32 - 1, dtype=np.int64) if cfg['seed'] == None else cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'running with seed: {seed}.')

    # create datasets
    train_dataset = Denoising_Dataset(cfg['train_dataset_path'], cfg, device, is_validation=False)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)

    validation_dataset = Denoising_Dataset(cfg['validation_dataset_path'], cfg, device, is_validation=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg['batch_size'], shuffle=True)

    # define model, loss function, optimizer and scheduler
    model = Model(model_type= cfg['model_type'], in_channels=1, n_classes=1)
    criterion = LossG(cfg)
    optimizer = get_optimizer(cfg, model.parameters())
    scheduler = get_scheduler(optimizer, cfg)

    # resume exists checkpoint, if needed
    resume_path = cfg['resume_path'] or f"pretrained_models/{cfg['model_name']}/{cfg['model_name']}-latest.pt"
    if os.path.exists(resume_path) and cfg['model_name'] != 'delete':
        checkpoint = torch.load(resume_path,  map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['step']
        optimal_validation_loss = checkpoint['optimal_validation_loss']
        print('loading ckpt from {}'.format(resume_path))
    else:
        optimal_validation_loss = math.inf
        current_epoch = 0
        
    # training process
    for _ in range(1, cfg['num_batches'] + 1):
        current_time = time.time()
        if current_time - start_time > cfg['running_time'] * 60 * 60:
            save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, save_name='latest')
            break
        current_epoch += 1
        raw_images, clean_images = next(iter(train_dataloader))
        raw_images, clean_images = raw_images.to(device), clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(raw_images)
        losses = criterion(outputs, clean_images)
        loss_G = losses['loss']
        log_data = {**losses, 'epoch': current_epoch}

        # validation set
        if current_epoch % cfg['validation_frequency'] == 0:
            with torch.no_grad():
                model.eval()
                validation_raw_images, validation_clean_images = next(iter(validation_dataloader))
                validation_raw_images, validation_clean_images = validation_raw_images.to(device), validation_clean_images.to(device)
                validation_outputs = model(validation_raw_images)
                validation_loss = criterion(validation_outputs, validation_clean_images)
                log_data['Validation loss'] = validation_loss['loss']
                model.train()
            if (validation_loss['loss'] < optimal_validation_loss) and current_epoch > 1000:
                save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='optimal_{}'.format(current_epoch))
                optimal_validation_loss = validation_loss['loss']

        # reshape and create binary masks
        b, c = clean_images.shape[:2]
        target = (clean_images > 0).float().reshape(b, c, -1)
        pred = (outputs['preds'].detach() > 0.5).float().reshape(b, c, -1)
        if current_epoch % cfg['validation_frequency'] == 0:
            validation_target = (validation_clean_images > 0).float().reshape(b, c, -1)
            validation_pred = (validation_outputs['preds'].detach() > 0.5).float().reshape(b, c, -1)
        # calculate F1 scores for evaluation
        rand_idx = np.random.randint(0, target.shape[0])
        target_i = target[rand_idx]
        pred_i = pred[rand_idx]
        f1 = (2 * (target_i[0] * pred_i[0]).sum() + epsilon) / (target_i[0].sum() + pred_i[0].sum() + epsilon)
        log_data[f'Training_f1'] = f1
        if current_epoch % cfg['validation_frequency'] == 0:
            validation_target_i = validation_target[rand_idx]
            validation_pred_i = validation_pred[rand_idx]
            f1 = (2 * (validation_target_i[0] * validation_pred_i[0]).sum() + epsilon) / (validation_target_i[0].sum() + validation_pred_i[0].sum() + epsilon)
            log_data[f'Validation_f1'] = f1
            
        # update learning rate
        lr = optimizer.param_groups[0]['lr']
        log_data["lr"] = lr

        # save checkpoint
        if current_epoch % cfg['save_checkpoint_freq'] == 0:
            save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name='latest')
            save_checkpoint(model, optimizer, current_epoch, optimal_validation_loss, outputs_dir, model_name, save_name=f'{current_epoch}')

        wandb.log(log_data)
        loss_G.backward()

        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    train_model()
