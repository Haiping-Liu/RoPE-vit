import argparse
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import models_v2_rope   
from engine import train_one_epoch_cor, evaluate
import utils
from datasets import build_coord_dataset  


def hf_checkpoint_load(model_name):
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="naver-ai/" + model_name, filename="pytorch_model.bin")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    return checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser('Coordinate prediction training script', add_help=False)

    # === Dataset ===
    parser.add_argument('--data-path', default='./synthetic_dataset', type=str)
    parser.add_argument('--nb-classes', default=6, type=int, help='Number of coordinate outputs')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    # === Model ===
    parser.add_argument('--model', default='rope_axial_coord_predictor_tiny', type=str,
                        help='Model name registered to timm')
    parser.add_argument('--backbone', default='rope_axial_deit_small_patch16_LS', type=str,
                        help='Backbone name if loading pretrained')
    parser.add_argument('--input-size', default=224, type=int, help='Input image size')
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop-path', type=float, default=0.1)

    # === Training control ===
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='./output_coord')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Only evaluate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    
    # === Optimizer ===
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt-eps', default=1e-8, type=float)
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip-grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.05)

    # === LR Scheduler ===
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-lr', type=float, default=1e-6)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--decay-epochs', type=int, default=30)
    parser.add_argument('--decay-rate', type=float, default=0.1)
    parser.add_argument('--cooldown-epochs', type=int, default=10)
    parser.add_argument('--patience-epochs', type=int, default=10)

    # === EMA ===
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996)
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)

    # === System ===
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser



def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    train_dataset = build_coord_dataset(is_train=True, args=args)
    val_dataset = build_coord_dataset(is_train=False, args=args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=4, pin_memory=True)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=6
    )

    if args.pretrained:
        checkpoint = hf_checkpoint_load(args.backbone)
        state_dict = checkpoint['model']

        # 删除 head 层参数（分类用的）
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}

        missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
        print(f"[Pretrained] Loaded with missing keys: {missing}, unexpected keys: {unexpected}")


    model.to(device)

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.MSELoss()
    model_ema = ModelEma(model, decay=0.99996, device='cpu')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_metric = float('inf')

    for epoch in range(args.epochs):
        train_stats = train_one_epoch_cor(
            model, criterion, train_loader, optimizer, device, epoch,
            loss_scaler=loss_scaler, model_ema=model_ema, args=args
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(val_loader, model, device)

        print(f"Epoch {epoch}: Val MSE {test_stats['loss']:.4f}")

        if test_stats['loss'] < max_metric:
            max_metric = test_stats['loss']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')

        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in test_stats.items()},
        }

        with open(output_dir / "log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print("Training time:", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Coordinate prediction', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

