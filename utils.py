# Rewritten based on MTI-Net and ATRC by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import shutil
from torchmetrics.functional.audio import signal_noise_ratio
import nvidia_smi
import torch.nn as nn
from torch.nn import init
import cv2
import torch.utils.data


def forward_pass(model, batch, prompt_mode, device):
    if prompt_mode == 'point':
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_points=batch["input_points"].to(device),
                        multimask_output=False)
    elif prompt_mode == 'box':    
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)
    elif prompt_mode == 'point+box':
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_points=batch["input_points"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)
    else:
        raise NotImplementedError
    return outputs
        

def get_bounding_box_center(ground_truth_map, bbox_threshold):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if (len(y_indices) == 0) or (len(x_indices) == 0):
        bbox = [0, 0, 0, 0]
        center = [0, 0]
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        #x_min = max(0, x_min - np.random.randint(0, bbox_threshold))
        #x_max = min(W, x_max + np.random.randint(0, bbox_threshold))
        #y_min = max(0, y_min - np.random.randint(0, bbox_threshold))
        #y_max = min(H, y_max + np.random.randint(0, bbox_threshold))

        x_min = max(0, x_min - bbox_threshold) # use fixed enlarge
        x_max = min(W, x_max + bbox_threshold)
        y_min = max(0, y_min - bbox_threshold)
        y_max = min(H, y_max + bbox_threshold)

        bbox = [x_min, y_min, x_max, y_max]
        center = [(x_min + x_max)/2, (y_min + y_max)/2]

    return bbox, center

    
def find_contour(img):
    img_numpy = img.numpy().astype(np.uint8)
    contours, _ = cv2.findContours(img_numpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    allseg = []
    if len(contours) < 2:
        return []
    else:
        for cont in contours:
            seg_in = np.zeros_like(img_numpy).astype(np.uint8)
            seg_in = np.ascontiguousarray(seg_in, dtype=np.uint8)
            cv2.fillPoly(seg_in, pts =[cont], color=(1,1,1))
            seg_out = seg_in.copy()
            allseg.append(torch.from_numpy(seg_out).to(torch.float))
    return allseg


class SAMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, bbox_threshold):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.bbox_threshold = bbox_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # get bounding box prompt
        bboxes, centers = get_bounding_box_center(ground_truth_mask, 
                                           bbox_threshold = self.bbox_threshold)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[[centers]], 
                                input_boxes=[[bboxes]],
                                return_tensors="pt")
        
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        inputs["bbox"] = torch.tensor(bboxes, dtype=torch.float32) # save for real bbox which may include enlarge pixels

        return inputs
    
    
def init_weights(net, init_type='normal', mean = 0.0, init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        # print(classname)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, mean)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, mean)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
    
def get_gpuinfo(gpu_id):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)

    gpu_temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
    fan_speed = nvidia_smi.nvmlDeviceGetFanSpeed(handle)
    gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
    memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    memory_usage = memory_info.used, memory_info.total
    return {'gpu_temp': gpu_temp, 'fan_speed': fan_speed, 
            'gpu_usage': gpu_usage, 'memory_used': memory_info.used,
            'memory_total': memory_info.total}

def get_loss(type):
    if type == 'l1':
        return nn.L1Loss()
    elif type == 'l2':
        return nn.MSELoss()
    elif type == 'ce':
        return nn.CrossEntropyLoss()
    elif type == 'bce':
        return nn.BCELoss()
    elif type == 'bcelogits':
        return nn.BCEWithLogitsLoss()
    else:
        return NotImplementedError
    
def set_optimizer(model, lr, mode):
    if mode == 'sgd':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif mode == "rmsprop":
        return torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif mode == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif mode == "adamw":
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif mode == "lbfgs":
        return torch.optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr)
    else:
        return NotImplementedError
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)
        
        
def computeSNR(x, xhat): 
    return signal_noise_ratio(x.flatten(), xhat.flatten())

def computeNRMS(x, xhat):
    if torch.norm(xhat.flatten()) == 0:
        return torch.zeros(1, dtype=xhat.dtype)
    else:
        return torch.norm(x.flatten() - xhat.flatten()) / (torch.norm(xhat.flatten()) + 1e-6)

def computerecall(x, xhat):
    if torch.sum(xhat) == 0:
        return torch.ones(1, dtype=xhat.dtype)
    else:
        return torch.sum(torch.multiply(x, xhat)) / (torch.sum(xhat) + 1e-6)

def center_cut_194(x):
    len_z = x.shape[-1]
    center = int(len_z/2)
    return x[:,:,center-97:center+97] # 3:197
    
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            pass

def load_and_freeze_encoder(model, checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, 'checkpoint', 'best_checkpoint.pytorch')
    model.load_state_dict(torch.load(checkpoint_path))
    print(f'Successfully load checkpoint from {checkpoint_path}')
    for param in model.encoders.parameters():
        param.requires_grad = False
    print('Setting encoders requires_grad to False')
    return model

def init_env(seed_value=42):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

def get_output(output, task):
    """Borrow from MTI-Net"""
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def to_cuda(batch):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) == list:
        return [to_cuda(v) for v in batch]
    else:
        return batch

def copytree_code(src_path, save_path):
    max_code_save = 100
    for i in range(max_code_save):
        code_path = save_path + 'code%d/' % i
        if not os.path.exists(code_path):
            shutil.copytree(src=src_path, dst=code_path)
            break
        
# From PyTorch internals
import collections.abc as container_abcs
from itertools import repeat
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

