# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import wandb


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 记录训练过程中的loss和lr
        if utils.is_main_process():
            wandb.log({
                "train/loss": loss_value,
                "train/learning_rate": optimizer.param_groups[0]["lr"]
            }, step=epoch * len(data_loader) + metric_logger.step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_depth(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler=None, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)  
        targets = targets.to(device, non_blocking=True)  

        outputs = model(samples) 
        outputs = torch.nn.functional.interpolate(outputs.clone(), size=targets.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 记录训练过程中的loss和lr
        if utils.is_main_process():
            wandb.log({
                "train/loss": loss_value,
                "train/learning_rate": optimizer.param_groups[0]["lr"]
            }, step=epoch * len(data_loader) + metric_logger.step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_depth(data_loader, model, device):
    criterion = torch.nn.MSELoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # 添加新的评估指标
    abs_rel = 0
    sq_rel = 0
    rmse = 0
    rmse_log = 0
    delta_1 = 0
    total_pixels = 0
    
    model.eval()
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs.clone(), size=targets.shape[2:], mode='bilinear', align_corners=False)
            
            # 计算各种评估指标
            pred = outputs
            gt = targets
            
            # 避免除以0
            mask = gt > 0
            
            # AbsRel
            abs_rel += torch.sum(torch.abs(pred[mask] - gt[mask]) / gt[mask]).item()
            
            # SqRel
            sq_rel += torch.sum(((pred[mask] - gt[mask]) ** 2) / gt[mask]).item()
            
            # RMSE
            rmse += torch.sum((pred[mask] - gt[mask]) ** 2).item()
            
            # RMSE log
            rmse_log += torch.sum((torch.log(pred[mask]) - torch.log(gt[mask])) ** 2).item()
            
            # δ < 1.25
            max_ratio = torch.max(pred[mask] / gt[mask], gt[mask] / pred[mask])
            delta_1 += torch.sum(max_ratio < 1.25).item()
            
            total_pixels += torch.sum(mask).item()
            
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())
    
    # 计算平均值
    abs_rel = abs_rel / total_pixels
    sq_rel = sq_rel / total_pixels
    rmse = torch.sqrt(torch.tensor(rmse / total_pixels))
    rmse_log = torch.sqrt(torch.tensor(rmse_log / total_pixels))
    delta_1 = delta_1 / total_pixels
    
    # 记录评估指标到wandb
    if utils.is_main_process():
        wandb.log({
            "test/abs_rel": abs_rel,
            "test/sq_rel": sq_rel,
            "test/rmse": rmse,
            "test/rmse_log": rmse_log,
            "test/delta_1": delta_1,
            "test/loss": metric_logger.loss.global_avg
        }, step=epoch)
    
    metric_logger.synchronize_between_processes()
    print('* AbsRel: {:.4f} SqRel: {:.4f} RMSE: {:.4f} RMSE log: {:.4f} δ<1.25: {:.4f}'.format(
        abs_rel, sq_rel, rmse, rmse_log, delta_1))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'delta_1': delta_1,
        'loss': metric_logger.loss.global_avg
    }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def visualize_depth(model, data_loader, device, epoch, save_dir='depth_visualization'):
    """
    可视化深度预测结果并上传到wandb
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(data_loader))
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        outputs = torch.nn.functional.interpolate(outputs.clone(), 
                                                size=targets.shape[2:], 
                                                mode='bilinear', 
                                                align_corners=False)
        
        # 创建图像网格
        plt.figure(figsize=(15, 5))
        
        # 显示输入图像
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(images[0].cpu().numpy(), (1, 2, 0)))
        plt.title('Input Image')
        plt.axis('off')
        
        # 显示预测深度
        plt.subplot(1, 3, 2)
        pred_depth = outputs[0].cpu().numpy()[0]
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
        plt.imshow(pred_depth, cmap='plasma')
        plt.title('Predicted Depth')
        plt.axis('off')
        
        # 显示真实深度
        plt.subplot(1, 3, 3)
        gt_depth = targets[0].cpu().numpy()[0]
        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        plt.imshow(gt_depth, cmap='plasma')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # 保存图像
        plt.suptitle(f'Epoch {epoch}')
        save_path = os.path.join(save_dir, f'depth_visualization_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
        
        # 上传到wandb
        if utils.is_main_process():
            wandb.log({
                "visualization/depth_prediction": wandb.Image(save_path)
            }, step=epoch)
        
        # 计算并打印评估指标
        mask = gt_depth > 0
        abs_rel = np.mean(np.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask])
        rmse = np.sqrt(np.mean((pred_depth[mask] - gt_depth[mask])**2))
        print(f'Visualization at Epoch {epoch}: AbsRel={abs_rel:.4f}, RMSE={rmse:.4f}')