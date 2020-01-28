import matplotlib
matplotlib.use('Agg')
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from data import get_eval_set
from functools import reduce
from segnet import segnet
from utils import check_mkdir, evaluate, confusion_matrix
import seaborn as sns
import time
import cv2
from math import log10
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# General parameters
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/datasets/task_test/')
parser.add_argument('--test_dir', type=str, default='test', help='Name of the test folder')
parser.add_argument('--output_dir', default='Results/', help='Location to save image results')
parser.add_argument('--num_classes', type=int, default=6, help='Number of semantic segmentation classes')
parser.add_argument('--models_dir', default='weights/', help='Location of trained models')
parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment')

# DBPN Parameters
parser.add_argument('--sr_model_name', type=str, default='DBPN', help='super resolution network name')
parser.add_argument('--sr_upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--sr_residual', type=bool, default=False)
parser.add_argument('--sr_model', default='models/DBPN_x8.pth', help='sr pretrained base model')

# Segnet parameters
parser.add_argument('--seg_model_name', type=str, default='segnet', help='semantic segmentation network name')
parser.add_argument('--seg_model', default='models/DBPN_x8.pth', help='semantic segementation pretrained base model')

opt = parser.parse_args()
gpus_list=range(opt.gpus)
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
    
    
def normalize_rows(array):
    sum = array.sum(axis=1)
    new = np.zeros(array.shape)
    for i in range(array.shape[0]):
        new[i] = array[i]/sum[i]
    return new


def test(test_loader, sr_model, seg_model):
    psnr_criterion = nn.MSELoss()
    if cuda:
        psnr_criterion = psnr_criterion.cuda(gpus_list[0])
        
    sr_model.eval()
    seg_model.eval()

    avg_psnr = 0
    avg_psnr_bicubic = 0
    msks_all, prds_all, = [], []
    
    for iteration, batch in enumerate(test_loader, 1):
        with torch.no_grad():
            lr, bicubic, img, msk, img_name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]
        print(img_name)
        if cuda:
            lr = lr.cuda(gpus_list[0])
            img = img.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])
            msk = msk.cuda(gpus_list[0])
        
        with torch.no_grad():
            sr_prediction = sr_model(lr)
                
            if opt.sr_residual:
                sr_prediction = sr_prediction + bicubic
            
            mse = psnr_criterion(sr_prediction, img)
            psnr = 10 * log10(1 / mse.data)
            avg_psnr += psnr
            mse = psnr_criterion(bicubic, img)
            psnr = 10 * log10(1 / mse.data)
            avg_psnr_bicubic += psnr
        
            seg_prediction = seg_model(sr_prediction)
            #seg_prediction = seg_model(sr_prediction*255)
        
            msk = msk.long()      
            
            seg_prediction = seg_prediction.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            
            msks_all.append(msk.squeeze_(0).cpu().numpy())
            prds_all.append(seg_prediction)
            
            tmp_path_seg = os.path.join('Results', exp_name, 'segmentation', img_name[0])
            tmp_path_sr = os.path.join('Results', exp_name, 'super-resolution', img_name[0])
            if 'grss' not in opt.data_dir:
                save_img_seg(seg_prediction, tmp_path_seg)
            else:
                save_img_seg(seg_prediction, tmp_path_seg, msk)
            save_img_sr(sr_prediction, tmp_path_sr)
            
    print('--------------------------------------------------------------------')
    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluate(prds_all, msks_all, opt.num_classes)
    print('Results: [PSNR %.4f], [acc %.4f], [acc_cls %.4f], [iou %.4f], [fwavacc %.4f], [kappa %.4f]' % 
        (avg_psnr/len(test_loader), acc, acc_cls, mean_iou, fwavacc, kappa))
    print('--------------------------------------------------------------------')
    
    heat_map(prds_all, msks_all)


def heat_map(prds_all, msks_all):
    if 'grss' in opt.data_dir:
        y_labels = ['Road', 'Tree', 'Red roof', 'Grey roof', 'Concrete\nroof', 'Vegetation']
        sr_heatmap = normalize_rows(confusion_matrix(prds_all, msks_all, opt.num_classes))
        
        fig = plt.figure(figsize=(6,6))
        ax = sns.heatmap(sr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'sr.png')
            
    elif 'coffee' in opt.data_dir:
        y_labels = ['non-coffee', 'coffee']
        sr_heatmap = normalize_rows(confusion_matrix(prds_all, msks_all, opt.num_classes))
        
        sns.set(font_scale=1.3) 
        fig = plt.figure(figsize=(3.5,3.5))
        ax = sns.heatmap(sr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'sr.png')
        
    elif 'vaihingen' in opt.data_dir or 'task_test' in opt.data_dir:
        y_labels = ['Impervious\nsurfaces', 'Building', 'Low\nvegetation', 'Tree', 'Car']
        sr_heatmap = normalize_rows(confusion_matrix(prds_all, msks_all, opt.num_classes))
        sr_heatmap = np.delete(sr_heatmap, -1, axis=0)
        sr_heatmap = np.delete(sr_heatmap, -1, axis=1)

        fig = plt.figure(figsize=(5,5))
        ax = sns.heatmap(sr_heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heat_maps/'+exp_name+'/'+'sr.png')
        


def save_img_sr(img, img_name):
    save_img = img.squeeze().clamp(0, 1).cpu().numpy().transpose(1,2,0)
    cv2.imwrite(img_name, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    
def save_img_seg(prds, tmp_path, label=None):
    h, w = prds.shape
    if 'grss' in opt.data_dir:
        label = label.squeeze_(0).cpu().numpy()
        new = np.zeros((h, w, 3), dtype=np.uint8)
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


def main():
    print('===> Loading datasets')
    test_set = get_eval_set(opt.data_dir, opt.test_dir, opt.sr_upscale_factor, opt.num_classes)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    print('Building SR model ', opt.sr_model_name)
    if opt.sr_model_name == 'DBPN':
        sr_model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.sr_upscale_factor) 
        sr_model = torch.nn.DataParallel(sr_model, device_ids=gpus_list)
        model_name = os.path.join(opt.models_dir, exp_name , opt.sr_model)
        print(model_name)
        sr_model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')
    else:
        sys.exit('Invalid SR network')

    print('Building SemSeg model', opt.seg_model_name)
    if opt.seg_model_name == 'segnet':
        seg_model = segnet(num_classes=opt.num_classes, in_channels=3)
        seg_model = torch.nn.DataParallel(seg_model, device_ids=gpus_list)
        model_name = os.path.join(opt.models_dir, exp_name, opt.seg_model)
        print(model_name)
        seg_model.load_state_dict(torch.load(model_name))
        print('Pre-trained SemSeg model is loaded.')
    else:
        sys.exit('Invalid Semantic segmentation network')
               
    if cuda:
        sr_model = sr_model.cuda(gpus_list[0])
        seg_model = seg_model.cuda(gpus_list[0])
    
    check_mkdir(os.path.join('Results'))
    check_mkdir(os.path.join('Results', exp_name))
    check_mkdir(os.path.join('Results', exp_name, 'segmentation'))
    check_mkdir(os.path.join('Results', exp_name, 'super-resolution'))
    check_mkdir(os.path.join('heat_maps'))
    check_mkdir(os.path.join('heat_maps', exp_name))
    
    test(test_loader, sr_model, seg_model)
    

if __name__ == '__main__':
    main()
