import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dbpn import Net as DBPN
from data import get_training_set, get_eval_set, get_validation_set
import socket
import time
from math import log10

import datetime
import random
import gc
import numpy as np
from skimage import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
from segnet import segnet
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
import cv2
import copy

# General parameters
parser = argparse.ArgumentParser(description='PyTorch Super Resolution and Semantic Segmentation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--epoch_num', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=501, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/datasets/task_test/')
parser.add_argument('--train_dir', type=str, default='train', help='Name of the training folder')
parser.add_argument('--val_dir', type=str, default=None, help='Validation dir')
#parser.add_argument('--test_dataset', type=str, help='Test dir')
parser.add_argument('--save_folder', type=str, default='weights/', help='Location to save checkpoint models')
parser.add_argument('--patch_size', type=int, default=480, help='Input images size')
parser.add_argument('--num_classes', type=int, default=6, help='Number of semantic segmentation classes')
parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
parser.add_argument('--alfa', type=float, default=1, help='Loss alfa value')
parser.add_argument('--beta', type=float, default=1, help='Loss beta value')

# DBPN Parameters
parser.add_argument('--sr_model_name', type=str, default='DBPN', help='super resolution network name')
parser.add_argument('--sr_upscale_factor', type=int, default=8, help='super resolution upscale factor')
parser.add_argument('--sr_lr', type=float, default=1e-4, help='SR network Learning Rate')
parser.add_argument('--sr_data_augmentation', type=bool, default=True)
parser.add_argument('--sr_residual', type=bool, default=False)
parser.add_argument('--sr_patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--sr_pretrained_model', type=str, help='sr pretrained base model')
parser.add_argument('--sr_pretrained', type=bool, default=False)

# Segnet parameters
parser.add_argument('--seg_model_name', type=str, default='segnet', help='semantic segmentation network name')
parser.add_argument('--seg_lr', type=float, default=1e-4, help='Semantic segmentation network Learning Rate')
parser.add_argument('--seg_weight_decay', type=float, default=5e-4, help='Semantic segmentation network weight decay')
parser.add_argument('--seg_momentum', type=float, default=0.9, help='Semantic segmentation network momentum')
parser.add_argument('--seg_lr_patience', type=int, default=100, help='Semantic segmentation network Learning Rate Patience')
parser.add_argument('--seg_pretrained_model', type=str, help='semantic segmentation pretrained base model')
parser.add_argument('--seg_pretrained', type=bool, default=False)

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

if opt.exp_name == '':
    if opt.data_dir[-1] == '/':
        opt.data_dir = opt.data_dir[:-1]
    exp_name = os.path.split(opt.data_dir)[1]+'_'+str(opt.sr_upscale_factor)
else:
    exp_name = opt.exp_name
print(exp_name)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    
    
def train(epoch, train_loader, sr_model, seg_model, sr_criterion, psnr_criterion,
          seg_criterion, sr_optimizer, seg_optimizer):
    epoch_loss = 0
    epoch_sr_loss = 0
    epoch_seg_loss = 0
    
    avg_psnr = 0
    
    msks_all, prds_all = [], []
    
    sr_model.train()
    seg_model.train()
    
    for iteration, batch in enumerate(train_loader, 1):
        lr, bicubic, img, msk, img_name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]
        if cuda:
            lr = lr.cuda(gpus_list[0])
            img = img.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])
            msk = msk.cuda(gpus_list[0])
            
        sr_optimizer.zero_grad()
        seg_optimizer.zero_grad()
        
        sr_prediction = sr_model(lr)
        
        if opt.sr_residual:
            sr_prediction = sr_prediction + bicubic
        sr_loss = sr_criterion(sr_prediction, img)
        
        seg_prediction = seg_model(sr_prediction)
        #seg_prediction = seg_model(sr_prediction*255)
        
        msk = msk.long()
        seg_loss = seg_criterion(seg_prediction, msk)       

        total_loss = opt.alfa*sr_loss + opt.beta*seg_loss
        
        total_loss.backward()
        sr_optimizer.step()
        seg_optimizer.step()
        
        with torch.no_grad():
            mse = psnr_criterion(sr_prediction, img)
            psnr = 10 * log10(1 / mse.data)
            avg_psnr += psnr
            seg_prediction = seg_prediction.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            epoch_sr_loss += sr_loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_loss += total_loss.item()
            msks_all.append(msk.squeeze_(0).cpu().numpy())
            prds_all.append(seg_prediction)
        
    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate(prds_all, msks_all, opt.num_classes)
    print('--------------------------------------------------------------------')
    print('TRN: [epoch %d], [loss %.4f], [sr_loss %.4f], [seg_loss %.4f], [PSNR %.4f], [acc %.4f], [acc_cls %.4f], [iou %.4f], [fwavacc %.4f], [kappa %.4f]' % 
        (epoch, epoch_loss/len(train_loader), epoch_sr_loss/len(train_loader), epoch_seg_loss/len(train_loader), avg_psnr/len(train_loader), 
        acc, acc_cls, mean_iou, fwavacc, kappa))
    #print('--------------------------------------------------------------------')
    

def validate(epoch, val_loader, sr_model, seg_model, sr_criterion, psnr_criterion, seg_criterion, sr_optimizer, seg_optimizer):
    sr_model.eval()
    seg_model.eval()
    
    epoch_loss = 0
    epoch_sr_loss = 0
    epoch_seg_loss = 0
    avg_psnr = 0
    avg_psnr_bicubic = 0
    msks_all, prds_all = [], []
    
    save_img_epoch = 50
    
    if epoch % save_img_epoch == 0:
        check_mkdir(os.path.join('outputs', exp_name, 'segmentation', 'epoch_' + str(epoch)))
        check_mkdir(os.path.join('outputs', exp_name, 'super-resolution', 'epoch_' + str(epoch)))
    
    for iteration, batch in enumerate(val_loader, 1):
        with torch.no_grad():
            lr, bicubic, img, msk, img_name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]
        if cuda:
            lr = lr.cuda(gpus_list[0])
            img = img.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])
            msk = msk.cuda(gpus_list[0])

        sr_optimizer.zero_grad()
        seg_optimizer.zero_grad()
        
        with torch.no_grad():
            sr_prediction = sr_model(lr)
                
            if opt.sr_residual:
                sr_prediction = sr_prediction + bicubic
                
            sr_loss = sr_criterion(sr_prediction, img)
            
            mse = psnr_criterion(sr_prediction, img)
            psnr = 10 * log10(1 / mse.data)
            avg_psnr += psnr
            mse = psnr_criterion(bicubic, img)
            psnr = 10 * log10(1 / mse.data)
            avg_psnr_bicubic += psnr
            
            seg_prediction = seg_model(sr_prediction)
            #seg_prediction = seg_model(sr_prediction*255)
            
            msk = msk.long()
            seg_loss = seg_criterion(seg_prediction, msk)       
            seg_prediction = seg_prediction.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            msks_all.append(msk.squeeze_(0).cpu().numpy())
            prds_all.append(seg_prediction)
            
            total_loss = opt.alfa*sr_loss + opt.beta*seg_loss
            
            epoch_sr_loss += sr_loss.data
            epoch_seg_loss += seg_loss.data
            epoch_loss += total_loss.data
            
            if epoch % save_img_epoch == 0:
                tmp_path_seg = os.path.join('outputs', exp_name, 'segmentation', 'epoch_{}'.format(epoch), img_name[0])
                tmp_path_sr = os.path.join('outputs', exp_name, 'super-resolution', 'epoch_{}'.format(epoch), img_name[0])
                if 'grss' in opt.data_dir:
                    save_img_seg(seg_prediction, tmp_path_seg, msk)
                else:
                    save_img_seg(seg_prediction, tmp_path_seg)
                save_img_sr(sr_prediction, tmp_path_sr)
                
        
    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate(prds_all, msks_all, opt.num_classes)
    #print('--------------------------------------------------------------------')
    print('VAL: [epoch %d], [loss %.4f], [sr_loss %.4f], [seg_loss %.4f], [PSNR %.4f], [acc %.4f], [acc_cls %.4f], [iou %.4f], [fwavacc %.4f], [kappa %.4f]' % 
        (epoch, epoch_loss/len(val_loader), epoch_sr_loss/len(val_loader), epoch_seg_loss/len(val_loader), avg_psnr/len(val_loader), 
        acc, acc_cls, mean_iou, fwavacc, kappa))
    print('Bicubic PSNR: %.4f' % (avg_psnr_bicubic/len(val_loader)))
    print('--------------------------------------------------------------------')
    return mean_iou


def checkpoint(epoch, sr_model, seg_model, postfix=''):
    if postfix == '':
        sr_model_out_path = os.path.join(opt.save_folder, exp_name, 'sr_'+exp_name+'_epoch_{}.pth'.format(epoch))
        seg_model_out_path = os.path.join(opt.save_folder, exp_name,'seg_'+exp_name+'_epoch_{}.pth'.format(epoch))
    else:
        sr_model_out_path = os.path.join(opt.save_folder, exp_name,'sr_'+exp_name+'_'+postfix+'.pth'.format(epoch))
        seg_model_out_path = os.path.join(opt.save_folder, exp_name,'seg_'+exp_name+'_'+postfix+'.pth'.format(epoch))
    torch.save(sr_model.state_dict(), sr_model_out_path)
    torch.save(seg_model.state_dict(), seg_model_out_path)
    print("Checkpoint saved to {} and {}".format(sr_model_out_path, seg_model_out_path))


def main():
    print('===> Loading datasets')
    train_set = get_training_set(opt.data_dir, opt.train_dir, opt.patch_size, opt.sr_patch_size, opt.sr_upscale_factor, 
        opt.num_classes, opt.sr_data_augmentation)
    
    if opt.val_dir != None:
        val_set = get_eval_set(opt.data_dir, opt.val_dir, opt.sr_upscale_factor, opt.num_classes)
        train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size)
        val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1)
    else:
        # Creating data indices for training and validation splits:
        validation_split = .2
        dataset_size = len(train_set)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.seed(opt.seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)    
        
        train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=1, sampler=val_sampler)
    
    
    
    print('Building SR model ', opt.sr_model_name)
    if opt.sr_model_name == 'DBPN':
        sr_model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.sr_upscale_factor) 
        sr_model = torch.nn.DataParallel(sr_model, device_ids=gpus_list)
        if opt.sr_pretrained:
            model_name = os.path.join(opt.save_folder + opt.sr_pretrained_model)
            print(model_name)
            sr_model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.')
    else:
        sys.exit('Invalid SR network')

    print('Building SemSeg model', opt.seg_model_name)
    
    if opt.seg_model_name == 'segnet':
        seg_model = segnet(num_classes=opt.num_classes, in_channels=3)
        if not opt.seg_pretrained:
            seg_model.init_vgg16_params()
            print('segnet params initialized')
            seg_model = torch.nn.DataParallel(seg_model, device_ids=gpus_list)
        if opt.seg_pretrained:
            model_name = os.path.join(opt.save_folder + opt.seg_pretrained_model)
            print(model_name)
            seg_model.load_state_dict(torch.load(model_name))
            print('Pre-trained SemSeg model is loaded.')
            seg_model = torch.nn.DataParallel(seg_model, device_ids=gpus_list)
            
    
    sr_criterion = nn.L1Loss()
    psnr_criterion = nn.MSELoss()
    if cuda:
        sr_model = sr_model.cuda(gpus_list[0])
        seg_model = seg_model.cuda(gpus_list[0])
        sr_criterion = sr_criterion.cuda(gpus_list[0])
        psnr_criterion = psnr_criterion.cuda(gpus_list[0])
    if 'grss' in opt.data_dir:
        seg_criterion = CrossEntropyLoss2d(ignore_index = -1).cuda()
    else:
        seg_criterion = CrossEntropyLoss2d().cuda()


    sr_optimizer = optim.Adam(sr_model.parameters(), lr=opt.sr_lr, betas=(0.9, 0.999), eps=1e-8)
    seg_optimizer = optim.Adam(seg_model.parameters(), lr=opt.seg_lr, weight_decay=opt.seg_weight_decay, 
                    betas=(opt.seg_momentum, 0.99))

    scheduler = ReduceLROnPlateau(seg_optimizer, 'min', factor=0.5, 
                    patience=opt.seg_lr_patience, min_lr=2.5e-5, verbose=True)
    
    check_mkdir(os.path.join('outputs', exp_name))
    check_mkdir(os.path.join('outputs', exp_name, 'segmentation'))
    check_mkdir(os.path.join('outputs', exp_name, 'super-resolution'))
    check_mkdir(os.path.join(opt.save_folder, exp_name))
    
    #best_iou = 0
    best_iou = val_results = validate(0, val_loader,
            sr_model, seg_model, 
            sr_criterion, psnr_criterion, seg_criterion,
            sr_optimizer, seg_optimizer)
    #sys.exit()
    #best_epoch = -1
    best_epoch = 0
    best_model = (sr_model, seg_model)
    since_last_best = 0

    for epoch in range(opt.start_iter, opt.epoch_num + 1):
        train(epoch, train_loader,  
            sr_model, seg_model, 
            sr_criterion, psnr_criterion, seg_criterion,
            sr_optimizer, seg_optimizer)
        
        val_results = validate(epoch, val_loader,
            sr_model, seg_model, 
            sr_criterion, psnr_criterion, seg_criterion,
            sr_optimizer, seg_optimizer)
        
        if val_results > best_iou:
            best_iou = val_results
            best_epoch = epoch
            print('New best iou ', best_iou)
            best_model = (copy.deepcopy(sr_model), copy.deepcopy(seg_model))
            since_last_best = 0
            checkpoint(epoch, sr_model, seg_model, 'tmp_best')
        else:
            print('Best iou epoch: ', best_epoch, ':', best_iou)
        
        scheduler.step(val_results)
        
        if (epoch) % (opt.epoch_num/2) == 0:
            for param_group in sr_optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('SR Learning rate decay: lr={}'.format(sr_optimizer.param_groups[0]['lr']))

        if (epoch) % (opt.snapshots) == 0:
            checkpoint(epoch, sr_model, seg_model)
            
        #since_last_best += 1
        #if since_last_best == 20:
        #    checkpoint(epoch, best_model[0], best_model[1], 'tmp_best')
            
    print('Saving final best model')
    checkpoint(epoch, best_model[0], best_model[1], 'best')
        

def save_img_sr(img, img_name):
    save_img = img.squeeze().clamp(0, 1).cpu().numpy().transpose(1,2,0)
    cv2.imwrite(img_name, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    
def save_img_seg(prds, tmp_path, label=None):
    h, w = prds.shape
    if 'grss' in opt.data_dir:
        new = np.zeros((h, w, 3), dtype=np.uint8)
        label = label.squeeze_(0).cpu().numpy()
        for i in range(h):
            for j in range(w):
                if label[i][j] == -1:
                    new[i][j] = [0,0,0]
                
                elif prds[i][j] == 0:
                    new[i][j] = [255,0,255]
                    
                elif prds[i][j] == 1:
                    new[i][j] = [0,255,0]
                    
                elif prds[i][j] == 2:
                    new[i][j] = [255,0,0]
                    
                elif prds[i][j] == 3:
                    new[i][j] = [0,255,255]
                    
                elif prds[i][j] == 4:
                    new[i][j] = [160,32,240]
                    
                elif prds[i][j] == 5:
                    new[i][j] = [46,139,87]
                    
                else:
                    sys.exit('Invalid prediction')
        
        io.imsave(tmp_path+'.png', new)
    elif 'coffee' in opt.data_dir:
        new = np.zeros((h, w), dtype=np.uint8)
                    
        for i in range(h):
            for j in range(w):
                if prds[i][j] == 0:
                    new[i][j] = 0
                    
                elif prds[i][j] == 1:
                    new[i][j] = 255
                    
                else:
                    sys.exit('Invalid prediction')
        
        io.imsave(tmp_path+'.png', new)
        
    elif 'vaihingen' in opt.data_dir or 'task_test' in opt.data_dir:
        new = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(h):
            for y in range(w):
                if prds[x][y] == 0:
                    new[x][y] = [255,255,255]
                    
                elif prds[x][y] == 1:
                    new[x][y] = [0,0,255]
                    
                elif prds[x][y] == 2:
                    new[x][y] = [0,255,255]
                    
                elif prds[x][y] == 3:
                    new[x][y] = [0,255,0]
                    
                elif prds[x][y] == 4:
                    new[x][y] = [255,255,0]
                    
                elif prds[x][y] == 5:
                    new[x][y] = [255,0,0]
                    
                else:
                    sys.exit('Invalid prediction')
        io.imsave(tmp_path+'.png', new)
    

if __name__ == '__main__':
    main()
