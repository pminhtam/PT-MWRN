import torch
import argparse
from model.mwrn_lv1 import MWRN_lv1
from torch.utils.data import DataLoader
from loss.loss import BasicLoss
import os
from data.data_provider import SingleLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
from model import common
import torch.nn as nn
def train(args):
    # torch.set_num_threads(4)
    # torch.manual_seed(args.seed)
    # checkpoint = utility.checkpoint(args)
    data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    loss_basic = BasicLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = MWRN_lv1().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 15, 20, 25, 30], 0.5)
    optimizer.zero_grad()
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint2 != "":
        if device == 'cuda':
            checkpoint2 = torch.load(args.checkpoint2)
        else:
            checkpoint2 = torch.load(args.checkpoint2, map_location=torch.device('cpu'))
        state_dict2 = checkpoint2['state_dict']
        model.lv2.load_state_dict(state_dict2)
        print("load lv2 done ....")
    try:
        checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_iter']
        best_loss = checkpoint['best_loss']
        state_dict = checkpoint['state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = "model."+ k  # remove `module.`
        #     new_state_dict[name] = v
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    except:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    DWT = common.DWT()
    param = [x for name, x in model.named_parameters()]
    clip_grad_D = 1e4
    grad_norm_D = 0
    for epoch in range(start_epoch, args.epoch):
        for step, (noise, gt) in enumerate(data_loader):
            noise = noise.to(device)
            gt = gt.to(device)
            x1 = DWT(gt).to(device)
            x2 = DWT(x1).to(device)
            x3 = DWT(x2).to(device)
            pred, img_lv2, img_lv3 = model(noise)
            # print(pred.size())
            loss_pred = loss_basic(pred, gt)
            scale_loss_lv2 = loss_basic(x2,img_lv2)
            scale_loss_lv3 = loss_basic(x3,img_lv3)
            loss = loss_pred+ scale_loss_lv2 + scale_loss_lv3
            optimizer.zero_grad()
            loss.backward()
            total_norm_D = nn.utils.clip_grad_norm_(param, clip_grad_D)
            grad_norm_D = (grad_norm_D*(step/(step+1)) + total_norm_D/(step+1))
            optimizer.step()
            average_loss.update(loss)
            if global_step % args.save_every == 0:
                print("Save : epoch ",epoch ," step : ", global_step," with avg loss : ",average_loss.get_value() , ",   best loss : ", best_loss )
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False
                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                print(global_step, "PSNR  : ", calculate_psnr(pred, gt))
                print(average_loss.get_value())
            global_step += 1
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        scheduler.step()
        print("Epoch : ", epoch , "end at step: ", global_step)

    # print(model)
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir', '-g' , default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz' , default=64, type=int, help='size of image')
    parser.add_argument('--epoch', '-e' ,default=1000, type=int, help='batch size')
    parser.add_argument('--batch_size','-bs' ,  default=2, type=int, help='batch size')
    parser.add_argument('--save_every','-se' , default=200, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le' , default=1, type=int, help='loss_every')
    parser.add_argument('--restart','-r' ,  action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint/lv1',
                        help='the checkpoint to eval')
    parser.add_argument('--checkpoint2', '-ckpt2', type=str, default='checkpoint/lv2/model_best.pth.tar',
                        help='the checkpoint lv 2')
    parser.add_argument('--color','-cl' , default=True, action='store_true')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')

    args = parser.parse_args()
    #
    train(args)
