import logging
import torch
import wandb

##############################################################################
### Many parts of this are a modified version of the official MoCo code ######
############### https://github.com/facebookresearch/moco #####################
##############################################################################
from tqdm import tqdm

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import logging_presets

from dataset_utils import batch_shuffle_single_gpu, batch_unshuffle_single_gpu, create_labels, create_dataset
from model import copy_params, create_encoder
from model_utils import knn_predict, update_records, save_model, test, train

logging.getLogger().setLevel(logging.INFO)

emb_dim = 128
n_workers = 16
batch_size = 512
lr = 0.03
paramK_momentum = 0.99
memory_size = 4096
num_epochs = 500
knn_k = 200
knn_t = 0.1
device = torch.device("cuda")
wandb.init(project="moco")
wandb.run.name = f"moco_{batch_size}"


(train_dataset, train_dataset_for_eval, val_dataset,
 train_loader, train_loader_for_eval, val_loader,) = create_dataset(batch_size, n_workers)

encQ = create_encoder(emb_dim, device)
# print(encQ)
encK = create_encoder(emb_dim, device)

# copy params from encQ into encK
copy_params(encQ, encK)

# optimizer = torch.optim.SGD(encQ.parameters(), lr, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(encQ.parameters(), lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

###########################################################
### Set the loss function and the (optional) miner here ###
############################################################
"""NTXentLoss = InfoNCE"""
loss_fn = losses.CrossBatchMemory(
    loss=losses.NTXentLoss(temperature=0.1), embedding_size=emb_dim, memory_size=memory_size)

dataset_dict = {"train": train_dataset_for_eval, "val": val_dataset}
record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)

# first check untrained performance
epoch = 0
best_accuracy = test(encQ, train_loader_for_eval, val_loader, epoch, knn_k, knn_t, record_keeper)

global_iteration = {"iter": 0}
for epoch in range(1, num_epochs + 1):
    logging.info("Starting epoch {}".format(epoch))
    loss = train(encQ, encK, paramK_momentum, loss_fn, optimizer, train_loader, record_keeper, global_iteration, device)
    curr_accuracy = test(
        encQ, train_loader_for_eval, val_loader, epoch, knn_k, knn_t, record_keeper
    )
    wandb.log({'curr_accuracy': curr_accuracy, 'train_loss': loss})
    if curr_accuracy > best_accuracy:
        best_accuracy = curr_accuracy
        save_model(encQ)
    scheduler.step()
