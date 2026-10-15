import os
import torch 
import numpy as np
from utils import init_env, find_contour
from julia import Julia
jl = Julia(compiled_modules=False)
from julia.AVSfldIO import fld_read, fld_write
from prepare_dataset_zhonglin import zhonglindataloader
    
if __name__ == "__main__":
    loader = zhonglindataloader(datasetdir='../y90-data-wden/zhonglin/test_data.mat', delete_zeros=False)
    result_path = '../result/test-sam-zhonglin-bbox=10-prompt=box'
    outputseg = fld_read(os.path.join(result_path, 'outputseg.fld'))
    trueseg = fld_read(os.path.join(result_path, 'trueseg.fld'))
    loader.meta['seg'] = loader.meta['seg'].permute(0,3,2,1).reshape(-1,loader.imgsize,loader.imgsize)
    
    put_back_idx = loader.meta['seg'].shape[0]
    for imgidx in range(loader.meta['seg'].shape[0]):
        seg = loader.meta['seg'][imgidx, :, :]
        filled_segs = find_contour(seg)
        if len(filled_segs) > 0:
            print(f'put {put_back_idx} back into {imgidx}')
            for j in range(1, len(filled_segs)):
                outputseg[imgidx] += outputseg[put_back_idx]
                trueseg[imgidx] += trueseg[put_back_idx]
                put_back_idx += 1
    outputseg = outputseg[:loader.meta['seg'].shape[0], :, :]
    trueseg = trueseg[:loader.meta['seg'].shape[0], :, :]
    print('outputseg shape: ', outputseg.shape)
    print('trueseg shape: ', trueseg.shape)
    fld_write(os.path.join(result_path, 'outputseg_putback.fld'), outputseg)
    fld_write(os.path.join(result_path, 'trueseg_putback.fld'), trueseg)
    
    
    