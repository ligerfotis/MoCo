import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd
import torch
import wandb

##############################################################################
### Many parts of this are a modified version of the official MoCo code ######
############### https://github.com/facebookresearch/moco #####################
##############################################################################
from tqdm import tqdm

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import logging_presets

from dataset_utils import create_dataset
from model import copy_params, create_encoder, ModelMoCo
from model_utils import save_model, test, train

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--num-workers', default=16, type=int, metavar='N', help='number of workers')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

'''
args = parser.parse_args()  # running in command line
'''
args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 500
args.batch_size = 256

args.cos = True
args.schedule = []  # cos in use
args.symmetric = False
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

print(args)
use_wandb = True
if use_wandb:
    device = torch.device("cuda")
    wandb.init(project="moco")
    wandb.run.name = f"moco_{args.batch_size}_{args.moco_k}"
    wandb.config.update(args)

(train_data, train_loader, memory_data, memory_loader, test_data, test_loader) = create_dataset(args)

# create model
model = ModelMoCo(
    dim=args.moco_dim,
    K=args.moco_k,
    m=args.moco_m,
    T=args.moco_t,
    arch=args.arch,
    bn_splits=args.bn_splits,
    symmetric=args.symmetric,
).cuda()

# print(model.encoder_q)

# encQ = create_encoder(args.emb_dim, device)
# # print(encQ)
# encK = create_encoder(args.emb_dim, device)

# copy params from encQ into encK
# copy_params(encQ, encK)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
# optimizer = torch.optim.Adam(encQ.parameters(), args.lr, weight_decay=5e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

###########################################################
### Set the loss function and the (optional) miner here ###
############################################################
"""NTXentLoss = InfoNCE"""
# loss_fn = losses.CrossBatchMemory(
#     loss=losses.NTXentLoss(temperature=0.1), embedding_size=args.emb_dim, memory_size=args.memory_size)

dataset_dict = {"train": memory_data, "val": test_data}
record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)

# define optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

# load model if resume
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))

# logging
results = {'train_loss': [], 'test_acc@1': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

# training loop
for epoch in range(epoch_start, args.epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch, optimizer, args)
    results['train_loss'].append(train_loss)
    test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
    results['test_acc@1'].append(test_acc_1)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    # save model
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
    if use_wandb:
        wandb.log({'test accuracy': test_acc_1, 'train loss': train_loss})