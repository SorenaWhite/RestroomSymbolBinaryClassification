import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch import optim as optim
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
from dataset.MMLRestroomSign import MMLRestroomSign, build_transform
from transformer_decoder import TransformerDecoder


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_args():
    parser = argparse.ArgumentParser('MobileMLP+LLM', add_help=False)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=bool, default=True)
    parser.add_argument('--data_root', default='/media/data/', type=str, help='dataset root')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR', help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--enable_wandb', action='store_true', help="enable logging to Weights and Biases")

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        print(args)
        self.epoch = args.epoch
        self.device = torch.device(args.device)
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        dataset_train = MMLRestroomSign(args.data_root, build_transform(args, is_train=True), self.device, is_train=True)
        dataset_val = MMLRestroomSign(args.data_root, build_transform(args, is_train=False), self.device, is_train=False)

        self.data_loader_train = DataLoader(
            dataset_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            shuffle=True
        )
        self.data_loader_val = DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )
        self.model = TransformerDecoder(num_classes=2)

        self.optimizer = optim.Adam(self.model.parameters())

        num_training_steps_per_epoch = len(dataset_train)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)


    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss_value = AverageMeter()
        acc_value = AverageMeter()
        for data_iter, data in enumerate(tqdm(self.data_loader_train)):
            image_tensor, text_tensor, target_tensor = data
            image_tensor = image_tensor.to(self.device)
            text_tensor = text_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            preds = self.model(image_tensor, text_tensor)

            loss = self.criterion(preds, target_tensor)

            loss_value.update(loss.item())
            acc_value.update(accuracy(preds, target_tensor))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            torch.cuda.synchronize()

        self.lr_scheduler.step()

        print(f"[Train] Loss: {loss_value.avg}, Acc: {acc_value.avg}")
        loss_value.reset()
        acc_value.reset()

    @torch.no_grad()
    def eval_one_epoch(self):
        criterion = torch.nn.CrossEntropyLoss()

        loss_value = AverageMeter()
        acc_value = AverageMeter()

        self.model.eval()
        for data_iter, data in enumerate(tqdm(self.data_loader_val)):
            image_tensor, text_tensor, target_tensor = data
            image_tensor = image_tensor.to(self.device)
            text_tensor = text_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            preds = self.model(image_tensor, text_tensor)

            loss = criterion(preds, target_tensor)

            loss_value.update(loss.item())
            acc_value.update(accuracy(preds, target_tensor))

        print(f"[Eval] Loss: {loss_value.avg}, Acc: {acc_value.avg}")
        loss_value.reset()
        acc_value.reset()

    def train(self):
        for epoch in range(self.epoch):
            print(f"------- Epoch {epoch} ----------")
            self.train_one_epoch()
            self.eval_one_epoch()


def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
