# prepare_dataset_zhonglin.py
# prepare data for zhonglin dataset
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import datasets
from grid_sample import *
import matplotlib.patches as patches
from utils import init_env, find_contour, get_bounding_box_center
import transcript

# import matlab.engine
# eng = matlab.engine.start_matlab()

class zhonglindataloader:
    def __init__(self, datasetdir = '.',
                 known_xtrue = True, 
                 verbose = True,
                 delete_zeros = True,
                 delete_fraction = 0.3) -> None:
        # torch.set_float32_matmul_precision('highest')
        self.imgsize = 256
        self.verbose = verbose
        self.datasetdir = datasetdir
        self.known_xtrue = known_xtrue
        self.delete_zeros = delete_zeros
        self.delete_fraction = delete_fraction
        self.split_idx = []          
        if self.known_xtrue:
            self.meta = {'spect': [], 'seg':[]}
        else:
            self.meta = {'spect': []}
        self.readmatfiles()
        # normalize xtrue
    def convert_data(self):
        for i in range(self.meta['spect'].shape[0]):
            maxofspect = torch.max(self.meta['spect'][i,:,:,:])
            if self.verbose:
                print('max of spect: ', maxofspect)
            max_of_spect = torch.max(self.meta['spect'][i,:,:,:])
            self.meta['spect'][i,:,:,:] = self.meta['spect'][i,:,:,:] / max_of_spect * 255.0
            if self.verbose:
                print('max of spect after normalization: ', torch.max(self.meta['spect'][i,:,:,:]))     
        
        # num of patient, imgsize, imgsize, slices -> (num of patient * slices) * imgsize * imgsize                         
        self.meta['spect'] = self.meta['spect'].permute(0,3,2,1).reshape(-1,self.imgsize,self.imgsize)

        spect_temp = torch.clone(self.meta['spect'])
        if self.known_xtrue:
            self.meta['seg'] = self.meta['seg'].permute(0,3,2,1).reshape(-1,self.imgsize,self.imgsize)
            seg_temp = torch.clone(self.meta['seg'])
            
            for imgidx in range(self.meta['seg'].shape[0]):
                seg = self.meta['seg'][imgidx, :, :]
                filled_segs = find_contour(seg)
                if len(filled_segs) > 0:
                    print(f'Slice {imgidx} in seg is separate into: {len(filled_segs)} slices')
                    seg_temp[imgidx, :, :] = filled_segs[0].to(self.meta['seg'].dtype)
                    for segidx in range(1, len(filled_segs)):
                        seg_temp = torch.cat((seg_temp, 
                                            filled_segs[segidx].unsqueeze(0).to(self.meta['seg'].dtype)),
                                            dim=0)
                        spect_temp = torch.cat((spect_temp, 
                                                self.meta['spect'][imgidx, :, :].unsqueeze(0)),
                                                dim=0)
                        self.split_idx.append([seg_temp.shape[0]-1, imgidx])
            self.meta['seg'] = torch.clone(seg_temp)
        
        self.meta['spect'] = torch.clone(spect_temp)
        
        if self.delete_zeros:
            assert self.known_xtrue == True
            spect_temp = torch.clone(self.meta['spect'])
            seg_temp = torch.clone(self.meta['seg'])
            spect_list_new = []
            seg_list_new = []
            for idx in range(spect_temp.shape[0]):
                if len(seg_temp[idx,:,:].nonzero()) == 0:
                    rand_number = torch.rand(1)
                    if rand_number < self.delete_fraction:
                        print(f'include idx: {idx}')
                        spect_list_new.append(spect_temp[idx,:,:])
                        seg_list_new.append(seg_temp[idx,:,:])
                else:
                    print(f'include idx: {idx}')
                    spect_list_new.append(spect_temp[idx,:,:])
                    seg_list_new.append(seg_temp[idx,:,:])
                    
            self.meta['spect'] = torch.stack(spect_list_new)
            self.meta['seg'] = torch.stack(seg_list_new)
                    
                    
        if self.verbose:
            print('spect shape: ', self.meta['spect'].shape)
            print('spect maximum: ', torch.max(self.meta['spect']))
            print('spect average: ', torch.mean(self.meta['spect']))
        if self.known_xtrue:
            if self.verbose:
                print('seg shape: ', self.meta['seg'].shape)
                print('seg maximum: ', torch.max(self.meta['seg']))
        
                
    def readmatfiles(self):
        spect = sio.loadmat(self.datasetdir)['spect']
        seg = sio.loadmat(self.datasetdir)['tumors']
        self.meta['spect'] = torch.from_numpy(spect)
        self.meta['seg'] = torch.from_numpy(seg)
        
                
def create_dataset(images, labels):
    dataset = datasets.Dataset.from_dict({"image": images,
                                "label": labels})
    dataset = dataset.cast_column("image", datasets.Image())
    dataset = dataset.cast_column("label", datasets.Image())

    return dataset


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    loader = zhonglindataloader(datasetdir='../y90-data-wden/zhonglin/test_data.mat', delete_zeros=False)
    loader.convert_data()
    init_env(seed_value=42)
    bbox_threshold = 10
    # path = './y90-data/test/y90res02'
    # spect, seg = read_mat_files(path=path)
    # spect = np.transpose(spect, [2,0,1]) / np.amax(spect) * 255.0
    # print('maximum of spect: ', np.amax(spect))
    # seg = np.transpose(seg, [2,0,1])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # num_of_img x dim_of_img
    dataset = create_dataset(images=loader.meta['spect'], labels=loader.meta['seg'])
    # processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
    # model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
    # plt.figure()
    transcript.start(f'tumor_location.log', mode='a')
    for idx in range(len(dataset)):
        ground_truth_seg = np.array(dataset[idx]["label"])
        bbox, _ = get_bounding_box_center(ground_truth_seg, bbox_threshold=10)
        _, center = get_bounding_box_center(ground_truth_seg, bbox_threshold=1)
        print(f'slice {idx}, tumor bbox: {bbox}, tumor center: {center}')
    for idx in range(len(loader.split_idx)):
        print(f'slice {loader.split_idx[idx][0]} is for slice {loader.split_idx[idx][1]}')
    transcript.stop()
    
    fig, axes = plt.subplots()
    idx = 114
    axes.imshow(np.array(dataset[idx]["image"]))
    ground_truth_seg = np.array(dataset[idx]["label"])
    bbox, _ = get_bounding_box_center(ground_truth_seg, bbox_threshold)
    _, center = get_bounding_box_center(ground_truth_seg, bbox_threshold=1)
    print('bbox values: ', bbox)
    print('center values: ', center)
    show_mask(ground_truth_seg, axes)
    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                             linewidth=3, edgecolor='r', facecolor='none')
    cent = patches.Rectangle((center[0], center[1]), 1, 1, 
                             linewidth=3, edgecolor='b', facecolor='none')
    # Add the patch to the Axes
    axes.add_patch(rect)
    axes.add_patch(cent)
    axes.title.set_text(f"Ground truth mask")
    axes.axis("off")
    plt.savefig('testimg.png')
    plt.show()
    # plt.figure()
    # plt.imshow(loader.meta['seg'][idx])
    # plt.savefig('testimg_seg.png')
    # plt.show()