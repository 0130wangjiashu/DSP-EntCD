import sys
import time
import numpy as np
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import torch
from utils.losses import change_loss
import os
import logging
import random
import wandb
from utils.utils import train_val
from utils.path_hyperparameter import ph
from stage1_models import DSP
from stage2_models import EntCD



def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
def train_net(dataset_name):
    """
    This is the workflow of training model and evaluating model,
    note that the dataset should be organized as
    :obj:`dataset_name`/`train` or `val`/`t1` or `t2` or `label`

    Parameter:
        dataset_name(str): name of dataset

    Return:
        return nothing
    """
    # 1. Create dataset, checkpoint and best model path
    train_dataset = BasicDataset(t1_images_dir=f'{dataset_name}/train/t1/',
                                 t2_images_dir=f'{dataset_name}/train/t2/',
                                 labels_dir=f'{dataset_name}/train/label/',
                                 train=True)
    val_dataset = BasicDataset(t1_images_dir=f'{dataset_name}/val/t1/',
                               t2_images_dir=f'{dataset_name}/val/t2/',
                               labels_dir=f'{dataset_name}/val/label/',
                               train=False)

    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                            batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # working device
    
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    log_wandb = wandb.init(project=ph.log_wandb_project, resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict,mode='offline')
    
    os.environ["WANDB_DIR"] = f"./{ph.log_wandb_project}"

    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Mixed Precision: {ph.amp}
    ''')

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things

    # net = DSP()
    net = EntCD(ph.cold_checkpoint)
    net = net.to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,
                                                      factor=ph.factor)  # learning rate scheduler
    grad_scaler = torch.amp.GradScaler('cuda')  # loss scaling for amp

    # load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate
    criterion = change_loss  # loss function

    best_metrics = {
    'best_f1score': 0,
    'lowest loss': float('inf')  # 设置为正无穷大，确保epoch_loss会比这个值小
    } # best evaluation metrics
    
    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    checkpoint_path = f'{ph.Train_path}A2Net_LEVIR+++_checkpoint/'
    best_f1score_model_path = f'{ph.Train_path}A2Net_LEVIR+++_best_f1score_model/'
    best_loss_model_path = f'{ph.Train_path}A2Net_LEVIR+++_best_loss_model/'

    non_improved_epoch = 0  # adjust learning rate when non_improved_epoch equal to patience

    # 5. Begin training
    print("--------------------模型开始训练------------------------")
    print(f'训练设备:',device)
    print(f'初始化学习率:',lr)
    for epoch in range(ph.epochs):

        log_wandb, net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
                optimizer=optimizer, scheduler=scheduler,total_step=total_step, lr=lr, criterion=criterion,
                to_pilimg=to_pilimg, epoch=epoch,warmup_lr=warmup_lr, grad_scaler=grad_scaler
            )

        # 6. Begin evaluation

        # starting validation from evaluate epoch to minimize time
        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                        optimizer=optimizer, scheduler=scheduler, total_step=total_step, lr=lr, criterion=criterion,
                        to_pilimg=to_pilimg, epoch=epoch,best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch
                    )

    wandb.finish()
    # os.system('shutdown')


if __name__ == '__main__':

    # set random seed to make the experiment reproducible
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.root_dir + ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)
