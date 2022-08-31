# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import copy
import numpy as np

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'


import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# from adamp import AdamP

from engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
import utils
import transforms as T


class LakesDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Lakes_png_images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Lakes_masks"))))

    def __getitem__(self, idx):
        
        
        
        # HORRIBLE HACK, PLEASE BE ASHAMED
        
        KeepLooping = True
        
        while KeepLooping :
            
            KeepLooping = False
            
            try :
                
                # load images and masks
                img_path = os.path.join(self.root, "Lakes_png_images", self.imgs[idx])
                mask_path = os.path.join(self.root, "Lakes_masks", self.masks[idx])
                
                # img = Image.open(img_path)
                img = Image.open(img_path).convert("RGB")
                
                # print('a')
                # print(img.size)
                
                # print(img_path)
                
                mask = Image.open(mask_path)
                mask = np.array(mask)

                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]
                        
                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                del mask

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)
                boxes = []
                
                # print(num_objs)
                
                innzero=[]
                
                for i in range(num_objs):
                    pos = np.where(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    
                    if ((xmax-xmin)*(ymax-ymin) != 0.):
                        boxes.append([xmin, ymin, xmax, ymax])
                        innzero.append(i)
                    
                    # print(i,(xmax-xmin)*(ymax-ymin))
                    # print(i,xmin,xmax,ymin,ymax)

                    
                del pos
                    
                # print(len(innzero))

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # there is only one class
                labels = torch.ones((num_objs,), dtype=torch.int64)
                masks = torch.as_tensor(masks[innzero,:,:], dtype=torch.uint8)
                
                # print(masks.shape)
                
                image_id = torch.tensor([idx])
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

                # suppose all instances are not crowd
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["masks"] = masks
                target["image_id"] = image_id
                target["area"] = area
                target["iscrowd"] = iscrowd

                if self.transforms is not None:
                    img, target = self.transforms(img, target)
            except IndexError :
                
                idx = (idx + 1) % len(self.imgs)
                
                KeepLooping = True

        return img, target

    def __len__(self):
        return len(self.imgs)

class LakesDataset_test(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Lakes_test"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Lakes_test", self.imgs[idx])
        # img = Image.open(img_path)
        img = Image.open(img_path).convert("RGB")
        
        target = img # fake target
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    
    # min_size=800, max_size=1333,
    # image_mean=None, image_std=None,
    # # RPN parameters
    # rpn_anchor_generator=None, rpn_head=None,
    # rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
    # rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
    # rpn_nms_thresh=0.7,
    # rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
    # rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
    # rpn_score_thresh=0.0,
    # # Box parameters
    # box_roi_pool=None, box_head=None, box_predictor=None,
    # box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
    # box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
    # box_batch_size_per_image=512, box_positive_fraction=0.25,
    # bbox_reg_weights=None,
    
    
    # rpn_nms_thresh = 0.7 # default
    # rpn_nms_thresh = 0.3
    
    # rpn_score_thresh=0.0 #default    
    # rpn_score_thresh=0.3    
    
    
    
    pretrained = True
    # pretrained = False
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained, trainable_backbone_layers=5)

    possible_backbone_names = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2'
    ]
    
    backbone_name = 'resnet152'

    backbone  = resnet_fpn_backbone(backbone_name, pretrained=pretrained)
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)


    '''

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256 # ???
    # and replace the mask predictor with a new one
    
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    '''

    for param in model.parameters():
        
        param.requires_grad = True        

    if not(pretrained) :

        for param in model.parameters():
            
            param =  2*torch.rand(param.size(), dtype=param.dtype, device=param.device, requires_grad=True) - 1





    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')

    # num_workers = 1
    # num_workers = 6
    num_workers = 2
    
    # our dataset has two classes only for now - background and lakes
    num_classes = 2
    # use our dataset and defined transformations
    root_name = './lakes_data'
    
    dataset = LakesDataset(root_name, get_transform(train=True))
    dataset_test = LakesDataset(root_name, get_transform(train=False))

    # load_split = True
    load_split = False

    if (load_split):
            
        eval_indices = np.loadtxt("eval_set.txt",dtype=int).tolist()
        load_indices = np.loadtxt("train_set.txt",dtype=int).tolist()
        
        n_eval = len(eval_indices)
        n_data = len(load_indices) + n_eval
        
    else:

        n_data = len(dataset)   
        n_eval = 0
        # n_eval = int(n_data / 5)
        # n_eval = int(n_data / 10)

        # split the dataset in train and test set
        indices = torch.randperm(n_data).tolist()
        
        eval_indices = copy.copy(indices[0:n_eval])
        eval_indices.sort()
        load_indices = copy.copy(indices[n_eval:n_data])
        load_indices.sort()
        
        print("Evaluation images : ")
        print(eval_indices)
        print('')
        
        np.savetxt("eval_set.txt", indices[0:n_eval], fmt='%d')
        np.savetxt("train_set.txt", indices[n_eval:n_data], fmt='%d')
    
    
    dataset_test = torch.utils.data.Subset(dataset_test, eval_indices)
    dataset = torch.utils.data.Subset(dataset, load_indices)
    
    print("Size of training set: ",len(dataset))
    print("Size of evaluation set: ",len(dataset_test))

    # batch_size = 1
    # batch_size = 2
    batch_size = 1


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn)
        
    # # get the model using our helper function
    # model = get_model_instance_segmentation(num_classes)

    # model = torch.load("./trainings/03_SGD_resnet152_before_bug/MyTraining_008.pt")
    # model = torch.load("./trainings/04_SGD_resnet152_before_second_bug/MyTraining_007.pt")
    model = torch.load("./Start.pt")


    # move model to the right device
    model.to(device)



    for param in model.parameters():
        param.requires_grad_(True)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # params = [p for p in model.parameters()]
    
    
    # print(len(params))
    tot_nparam = 0
    for i in range(len(params)):
        tot_nparam += int(params[i].numel())
        
    print(f'Total numbers of parameters : {tot_nparam}')
    
    # print(model)
    
    # print(1/0)


    # optimizer = torch.optim.SGD(params, lr=0.0001,momentum=0.9, weight_decay=0.005)
    # optimizer = torch.optim.SGD(params, lr=0.002,momentum=0.9, weight_decay=0)
    
    # optimizer = torch.optim.SGD(params, lr=0.003,momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(params, lr=0.0004,momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(params, lr=0.00004,momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(params, lr=0.000005,momentum=0.9, weight_decay=0)
    
    # optimizer = torch.optim.Adam(params, lr=0.003)
    # optimizer = torch.optim.Adam(params, lr=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.00003)
    
    
    # optimizer = torch.optim.NAdam(params, lr=0.00005)
    # optimizer = torch.optim.RAdam(params, lr=0.00005)
    # optimizer = AdamP(params, lr=0.00005)
    
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.8)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25,gamma=0.9)


    num_epochs = 200
    
    summary = SummaryWriter()

    for epoch in range(num_epochs):
        
        print_freq = 1000
        # print_freq = 300
        # print_freq = 10
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,summary)
        # update the learning rate
        lr_scheduler.step()
        
        summary.flush()
        
        save_freq = 1
        
        if ((epoch % save_freq) == 0):
            
            n_save = epoch//save_freq
            torch.save(model, "./MyTraining_"+str(n_save).zfill(3)+".pt")
            # torch.save(model, "./Restart_"+str(n_save).zfill(3)+".pt")
            
            
            
    # torch.save(model, "./MyTraining_final.pt")

    
    summary.close()


    print("That's it!")
    
if __name__ == "__main__":
    main()
