import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import colorsys
import random

from PIL import Image
import numpy as np
import transforms as T

import torchvision.transforms

from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt
from matplotlib import patches,  lines

def np_safe_copy(array_in,xi,xf,yi,yf):
    
    array_out = np.zeros((xf-xi,yf-yi),dtype=array_in.dtype)
    
    the_xi = max(0,xi)
    the_xf = min(array_in.shape[0],xf)
    
    the_yi = max(0,yi)
    the_yf = min(array_in.shape[1],yf)
    
    x_out_start = the_xi - xi
    x_out_end = the_xf - xi
    
    y_out_start = the_yi - yi
    y_out_end = the_yf - yi
    
    array_out[x_out_start:x_out_end,y_out_start:y_out_end] = array_in[the_xi:the_xf,the_yi:the_yf]
    
    return array_out


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,summary):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        
        summary.add_scalar("Loss/train", loss_value, epoch)
        for k, v in loss_dict_reduced.items():
            summary.add_scalar(k+"/train", v, epoch)
            
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate_old(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.no_grad()
# ~ def evaluate(model, data_loader, device,epoch, print_freq,summary):
def evaluate(model, data_loader, device, epoch, print_freq,summary):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        
        summary.add_scalar("Loss/eval", loss_value, epoch)
        for k, v in loss_dict_reduced.items():
            summary.add_scalar(k+"/eval", v, epoch)
            

            







            

@torch.no_grad()
def plot_model(model, data_loader, device,output_folder='./output/'):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    for idx_img,data in enumerate(data_loader):
        
        images  = data[0]
        
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        images = list(img.to(cpu_device) for img in images)

        for i in range(len(images)):
            
            img = images[i]
            out = outputs[i]
            
            img_color = torch.stack((img[0,:,:],)*3, axis=0)
            
            themask = out['masks']
            theboxes = out['boxes']

            # ~ print(img.shape)
            # ~ print(themask.dtype)
            # ~ print(themask.shape)
            # ~ print(torch.amax(themask))
            # ~ print(torch.amin(themask))
            # ~ print('')
            
            colors = random_colors(themask.shape[0])
            
            thresh = 0.5
            # ~ thresh = 0.1
            
            for i in range(themask.shape[0]):
                apply_mask(img_color,themask[i,:,:,:],color=colors[i],thresh=thresh)
            
            img_color = np.array(torchvision.transforms.ToPILImage()(img_color))
        
            my_dpi = 96
            img_sizes = np.shape(img_color) 
            fig = plt.figure()
            fig.set_size_inches(img_sizes[0]/my_dpi, img_sizes[1]/my_dpi, forward = False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(img_color)
            
            
            for i in range(themask.shape[0]):
                p = patches.Rectangle((theboxes[i,0], theboxes[i,1]), theboxes[i,2] - theboxes[i,0], theboxes[i,3] - theboxes[i,1], linewidth=2,
                      edgecolor=colors[i], facecolor='none')
                ax.add_patch(p)
            

            output_filename = output_folder+str(idx_img)+'.png'
                
            plt.savefig(output_filename,bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            

        
@torch.no_grad()
def get_one_mask_and_plot(model, data_loader, device,image_output_filename,thresh=0.5):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    if len(data_loader) > 1 :
        
        raise ValueError("Data loader should only have one image in make poly")
    
    for idx_img,data in enumerate(data_loader):
        
        images  = data[0]
        
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        images = list(img.to(cpu_device) for img in images)

        for i in range(len(images)):
            
            img = images[i]
            out = outputs[i]
            
            img_color = torch.stack((img[0,:,:],)*3, axis=0)
            
            themask = out['masks']
            theboxes = out['boxes']

            npoly = themask.shape[0]
            nx = themask.shape[2]
            ny = themask.shape[3]

            # ~ enhance = cpte_enhance(nx,ny,thresh=15,maxval=7)
            # ~ enhance = cpte_enhance(nx,ny,thresh=15,maxval=4,exponent=1)
            # ~ enhance = cpte_enhance(nx,ny,thresh=15,maxval=0)
            
            # ~ for ipoly in range(npoly):
                # ~ themask[ipoly,0,:,:] *= enhance
            
            # ~ colors = random_colors(themask.shape[0])
            

            lt_colors = random_colors(1)
            
            colors = []
            for ilbl in range(themask.shape[0]):
                colors.append(lt_colors[0])
            
            
            # Plot whatever
            
            for i in range(themask.shape[0]):
                apply_mask(img_color,themask[i,:,:,:],color=colors[i],thresh=thresh)
            
            img_color_PIL = torchvision.transforms.ToPILImage()(img_color)
            img_color_PIL.save(image_output_filename)
        
            return themask


        
@torch.no_grad()
def get_one_mask_no_plot(model, data_loader, device,thresh=0.5):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    if len(data_loader) > 1 :
        
        raise ValueError("Data loader should only have one image in make poly")
    
    for idx_img,data in enumerate(data_loader):
        
        images  = data[0]
        
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs = [{k: v for k, v in t.items()} for t in outputs]
        
        out = outputs[0]
        themask = out['masks']
        return themask
        
            


            

@torch.no_grad()
def plot_exact(data_loader,output_folder='./output/'):
    
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    for idx_img,data in enumerate(data_loader):
        
        images  = data[0]
        outputs = data[1]
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        images = list(img.to(cpu_device) for img in images)

        for i in range(len(images)):
            
            img = images[i]
            out = outputs[i]
            
            img_color = torch.stack((img[0,:,:],)*3, axis=0)
            
            themask = out['masks']
            theboxes = out['boxes']
            
            #only for WB data
            msk_shape = themask.shape
            themask = torch.reshape(themask, (msk_shape[0],1,msk_shape[1],msk_shape[2]))


            # ~ print(img.shape)
            # ~ print(themask.dtype)
            # ~ print(themask.shape)
            # ~ print(torch.amax(themask))
            # ~ print(torch.amin(themask))
            # ~ print('')
            
            colors = random_colors(themask.shape[0])
            
            
            for i in range(themask.shape[0]):
                apply_mask(img_color,themask[i,:,:,:],color=colors[i])
            
            img_color = np.array(torchvision.transforms.ToPILImage()(img_color))
        
            my_dpi = 96
            img_sizes = np.shape(img_color) 
            fig = plt.figure()
            fig.set_size_inches(img_sizes[0]/my_dpi, img_sizes[1]/my_dpi, forward = False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(img_color)
            
            
            for i in range(themask.shape[0]):
                p = patches.Rectangle((theboxes[i,0], theboxes[i,1]), theboxes[i,2] - theboxes[i,0], theboxes[i,3] - theboxes[i,1], linewidth=2,
                      edgecolor=colors[i], facecolor='none')
                ax.add_patch(p)

            output_filename = output_folder+str(idx_img)+'.png'
            plt.savefig(output_filename,bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            

            

 
 


def apply_mask(image, mask, color=(.0,.0,1.), alpha=.2,thresh=0.5):
    for c in range(3):
        image[c,:, :] = torch.where(mask > thresh,
                                  image[c,:, :] *(1 - alpha) + alpha * color[c],
                                  image[c,:, :])
    return image
    
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # ~ random.shuffle(colors)
    return colors


def cpte_enhance(nx,ny,thresh=15,maxval=10,exponent=2):
    
    cpu_device = torch.device("cpu")
    
    band = torch.zeros((thresh),device=cpu_device,dtype=torch.float32)
    for i in range(thresh):
        band[i] = maxval * abs((i-thresh)/thresh)**exponent
        
    revband = torch.zeros((thresh),device=cpu_device,dtype=torch.float32)
    for i in range(thresh):
        revband[i] = maxval * abs((i)/thresh)**exponent
    
    enhance = torch.ones((nx,ny),device=cpu_device,dtype=torch.float32)
    
    for ix in range(nx):
        enhance[ix,0:thresh] += band
    
    for ix in range(nx):
        enhance[ix,ny-thresh:ny] += revband
    
    for iy in range(ny):
        enhance[0:thresh,iy] += band
    
    for ix in range(nx):
        enhance[nx-thresh:nx,iy] += revband
    
    
    return enhance
    
