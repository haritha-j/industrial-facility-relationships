"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse
import torch.nn as nn

from pathlib import Path

from src.preparation import *
from src.dataset import *
from src.chamfer import get_chamfer_loss_from_param_tensor, get_correspondence_loss_from_param_tensor

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, 'pointnet2')
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='PointAttn', help='model name [default: PointAttn]')
    #parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: PointAttn]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, device, criterion, cat):
    losses = []
    predictor = model.eval()

    for j, data  in tqdm(enumerate(loader), total=len(loader)):
        points, target = data['pointcloud'].to(device).float(), data['properties'].to(device)

        points = points.transpose(2, 1)
        pred, _ = predictor(points)
        loss = criterion(pred, target)
        chamfer_scale = 0.0005
        #chamfer_loss = get_chamfer_loss_tensor(pred, points, cat) * chamfer_scale
        chamfer_loss = get_chamfer_loss_from_param_tensor(pred, target, cat) * chamfer_scale
        #correspondence_loss = get_correspondence_loss_from_param_tensor(pred, target, cat)
        #losses.append(chamfer_scale*correspondence_loss +loss)
        #losses.append(loss + chamfer_loss)
        losses.append(chamfer_loss)
        #losses.append(loss)

    avg_loss = sum(losses)/len(losses)
    return avg_loss


def _make_dir(exp_dir):
    try:
        exp_dir.mkdir()
    except:
        pass


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    path = Path('output/')
    #savepath = '/content/drive/MyDrive/ElementNet/'
    savepath = 'models/'
    train_transforms = transforms.Compose([
                    Normalize(),
#                    RandomNoise(),
                    ToTensor()
                    ])

    cat = 'flange'
    train_ds = PointCloudData(path, category=cat, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', category=cat, transform=train_transforms)
    targets = train_ds.targets
    trainDataLoader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=args.batch_size)
    test_criterion = nn.MSELoss()

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('pointnet2/log/')
    _make_dir(exp_dir)
    exp_dir = exp_dir.joinpath('classification')
    _make_dir(exp_dir)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    _make_dir(exp_dir)
    checkpoints_dir = exp_dir.joinpath('checkpoints_pointattn/')
    _make_dir(checkpoints_dir)
    log_dir = exp_dir.joinpath('logs/')
    _make_dir(log_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('pointnet2/models/%s.py' % args.model, str(exp_dir))
    shutil.copy('pointnet2/models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    predictor = model.get_model(targets, normal_channel=args.use_normals)
    criterion = model.get_loss()
    predictor.apply(inplace_relu)

    if not args.use_cpu:
        predictor = predictor.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints_pointattn/best_model.pth')
        start_epoch = checkpoint['epoch']
        predictor.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_loss = math.inf
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        predictor = predictor.train()

        for data in tqdm(trainDataLoader):
            optimizer.zero_grad()
            points, target = data['pointcloud'].to(device).float(), data['properties'].to(device)

            points = points.transpose(2, 1)

            pred, trans_feat = predictor(points)
            #print("pred", pred.shape, trans_feat.shape)
            loss = criterion(pred, target, trans_feat, points, cat)
            #print("loss", loss)

            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()
        log_string('Train loss: %f' % loss)

        with torch.no_grad():
            loss = test(predictor.eval(), testDataLoader, device, test_criterion, cat)

            if (loss <= best_loss):
                best_loss = loss
                best_epoch = epoch + 1

            log_string('Test loss: %f' % (loss))
            log_string('Best loss: %f' % (best_loss))

            if (loss <= best_loss):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'loss': loss,
                    'model_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
