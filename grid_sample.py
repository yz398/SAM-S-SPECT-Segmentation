# grid_sample.py
# pytorch interpolation of interp3 in matlab
import torch
import torch.nn as nn
import torch.nn.functional as F 
import scipy.io as sio
import numpy as np

def resize128to512(V):
    """resize 3d tensors to 512x512x194

    Args:
        V (_type_): has shape (D, H, W)
    """
    Y_OFF = 0
    X_OFF = 0
    Z_OFF = 0
    
    V = V.unsqueeze(0).unsqueeze(0)
    
    NX = 128
    NY = 128
    NZ = 80
    L_XY = 4.79520
    L_Z = 4.79520

    NXI = 512
    NYI = 512
    NZI = 194
    L_XYI = 0.976562
    L_ZI = 2.0000

    X = (torch.arange(NX) + 0.5) * L_XY - NX*L_XY/2 + X_OFF
    Y = (torch.arange(NY) + 0.5) * L_XY - NY*L_XY/2 + Y_OFF
    Z = (torch.arange(NZ) + 0.5) * L_Z - NZ*L_Z/2 + Z_OFF

    X_scale = X[-1]
    Y_scale = Y[-1]
    Z_scale = Z[-1]

    XI = (torch.arange(NXI) + 0.5) * L_XYI - NXI*L_XYI/2
    YI = (torch.arange(NYI) + 0.5) * L_XYI - NYI*L_XYI/2
    ZI = (torch.arange(NZI) + 0.5) * L_ZI - NZI*L_ZI/2
    
    # normalize grid    
    XI = XI / X_scale
    YI = YI / Y_scale
    ZI = ZI / Z_scale
    
    grid_x, grid_y, grid_z = torch.meshgrid(XI, YI, ZI, indexing='xy')
        
    grid = torch.stack((grid_z, grid_x, grid_y), dim=-1).unsqueeze(0)
    
    Vq = F.grid_sample(V, grid.to(V.device), padding_mode= 'zeros', align_corners=True).squeeze()

    return Vq


def resize512to128(V):
    """resize 3d tensors to 128x128x80

    Args:
        V (_type_): has shape (D, H, W)
    """
    Y_OFF = 0
    X_OFF = 0
    Z_OFF = 0
    
    V = V.unsqueeze(0).unsqueeze(0)
    
    NX = 512
    NY = 512
    NZ = 194
    L_XY = 0.976562
    L_Z = 2.0000

    NXI = 128
    NYI = 128
    NZI = 80
    L_XYI = 4.79520
    L_ZI = 4.79520

    X = (torch.arange(NX) + 0.5) * L_XY - NX*L_XY/2 + X_OFF
    Y = (torch.arange(NY) + 0.5) * L_XY - NY*L_XY/2 + Y_OFF
    Z = (torch.arange(NZ) + 0.5) * L_Z - NZ*L_Z/2 + Z_OFF

    X_scale = X[-1]
    Y_scale = Y[-1]
    Z_scale = Z[-1]

    XI = (torch.arange(NXI) + 0.5) * L_XYI - NXI*L_XYI/2
    YI = (torch.arange(NYI) + 0.5) * L_XYI - NYI*L_XYI/2
    ZI = (torch.arange(NZI) + 0.5) * L_ZI - NZI*L_ZI/2
    
    # normalize grid    
    XI = XI / X_scale
    YI = YI / Y_scale
    ZI = ZI / Z_scale
    
    grid_x, grid_y, grid_z = torch.meshgrid(XI, YI, ZI, indexing='xy')
        
    grid = torch.stack((grid_z, grid_x, grid_y), dim=-1).unsqueeze(0)
    
    Vq = F.grid_sample(V, grid.to(V.device), padding_mode= 'zeros', align_corners=True).squeeze()

    return Vq


def resize128to256(V):
    """resize 128x128x80 to 256x256x80

    Args:
        V (_type_): has shape (D, H, W)
    """
    Y_OFF = 0
    X_OFF = 0
    Z_OFF = 0
    
    V = V.unsqueeze(0).unsqueeze(0)
    
    NX = 128
    NY = 128
    NZ = 80
    L_XY = 4.79520
    L_Z = 4.79520

    NXI = 256
    NYI = 256
    NZI = 80
    L_XYI = 4.79520 / 2
    L_ZI = 4.79520

    X = (torch.arange(NX) + 0.5) * L_XY - NX*L_XY/2 + X_OFF
    Y = (torch.arange(NY) + 0.5) * L_XY - NY*L_XY/2 + Y_OFF
    Z = (torch.arange(NZ) + 0.5) * L_Z - NZ*L_Z/2 + Z_OFF

    X_scale = X[-1]
    Y_scale = Y[-1]
    Z_scale = Z[-1]

    XI = (torch.arange(NXI) + 0.5) * L_XYI - NXI*L_XYI/2
    YI = (torch.arange(NYI) + 0.5) * L_XYI - NYI*L_XYI/2
    ZI = (torch.arange(NZI) + 0.5) * L_ZI - NZI*L_ZI/2
    
    # normalize grid    
    XI = XI / X_scale
    YI = YI / Y_scale
    ZI = ZI / Z_scale
    
    grid_x, grid_y, grid_z = torch.meshgrid(XI, YI, ZI, indexing='xy')
        
    grid = torch.stack((grid_z, grid_x, grid_y), dim=-1).unsqueeze(0)
    
    Vq = F.grid_sample(V, grid.to(V.device), padding_mode= 'zeros', align_corners=True).squeeze()

    return Vq


def resize512to256(V):
    """resize 512x512x194 tensors to 256x256x80

    Args:
        V (_type_): has shape (D, H, W)
    """
    Y_OFF = 0
    X_OFF = 0
    Z_OFF = 0
    
    V = V.unsqueeze(0).unsqueeze(0)
    
    NX = 512
    NY = 512
    NZ = 194
    L_XY = 0.976562
    L_Z = 2.0000

    NXI = 256
    NYI = 256
    NZI = 80
    L_XYI = 4.79520 / 2
    L_ZI = 4.79520

    X = (torch.arange(NX) + 0.5) * L_XY - NX*L_XY/2 + X_OFF
    Y = (torch.arange(NY) + 0.5) * L_XY - NY*L_XY/2 + Y_OFF
    Z = (torch.arange(NZ) + 0.5) * L_Z - NZ*L_Z/2 + Z_OFF

    X_scale = X[-1]
    Y_scale = Y[-1]
    Z_scale = Z[-1]

    XI = (torch.arange(NXI) + 0.5) * L_XYI - NXI*L_XYI/2
    YI = (torch.arange(NYI) + 0.5) * L_XYI - NYI*L_XYI/2
    ZI = (torch.arange(NZI) + 0.5) * L_ZI - NZI*L_ZI/2
    
    # normalize grid    
    XI = XI / X_scale
    YI = YI / Y_scale
    ZI = ZI / Z_scale
    
    grid_x, grid_y, grid_z = torch.meshgrid(XI, YI, ZI, indexing='xy')
        
    grid = torch.stack((grid_z, grid_x, grid_y), dim=-1).unsqueeze(0)
    
    Vq = F.grid_sample(V, grid.to(V.device), padding_mode= 'zeros', align_corners=True).squeeze()

    return Vq

# if __name__ == '__main__':
#     V = torch.tensor(sio.loadmat('../misc/randomV.mat')['V'], dtype=torch.float)
#     Vqmatlab = sio.loadmat('../misc/randomVq.mat')['Vq']
#     Vq = resize128to256(V).numpy()
#     # sio.savemat('../misc/randomVq-python.mat', {'Vqpython': Vq})
#     # print(Vq[:,:,0] - Vqmatlab[:,:,0])
#     print('Vq-Vqmatlab norm diff: ', np.linalg.norm(Vqmatlab - Vq) / np.linalg.norm(Vqmatlab))
#     Vb = resize256to128(torch.from_numpy(Vq)).numpy()
#     # sio.savemat('../misc/randomVb-python.mat', {'Vb': Vb})
#     Vbmatlab = sio.loadmat('../misc/Vb-matlab.mat')['Vqb']
#     print('Vb-V norm diff: ', (np.linalg.norm(Vb - Vbmatlab) / np.linalg.norm(Vbmatlab)).item())
    # print('Vq shape: ', Vq.shape)