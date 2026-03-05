import numpy as np
import torch
import pywt

def dwt_matrix(size, wavelet_name='haar'):
    # size should be the size of the image changed the input of the function to the image
    # to get its sizes
    wavelet = pywt.Wavelet(wavelet_name)
    lo = np.array(wavelet.dec_lo)
    hi = np.array(wavelet.dec_hi)

    half = size // 2
    filter_len = len(lo)

    matrix_Low  = np.zeros((half, size))
    matrix_High = np.zeros((half, size))

    for i in range(half):
        for j in range(filter_len):
            idx = (2*i + j) % size
            matrix_Low[i, idx]  += lo[j]
            matrix_High[i, idx] += hi[j]

    matrix_Low  = torch.tensor(matrix_Low,  dtype=torch.float32)
    matrix_High = torch.tensor(matrix_High, dtype=torch.float32)
    return matrix_Low, matrix_High

def dwt(input, matrix_Low, matrix_High):
        L = torch.matmul(matrix_Low, input)
        H = torch.matmul(matrix_High, input)
        LL = torch.matmul(L, matrix_Low.t())
        LH = torch.matmul(L, matrix_High.t())
        HL = torch.matmul(H, matrix_Low.t())
        HH = torch.matmul(H, matrix_High.t())
        return LL, LH, HL, HH

def idwt_matrix(size, wavelet_name='haar'):
    wavelet = pywt.Wavelet(wavelet_name)
    lo = np.array(wavelet.rec_lo)[::-1] # need to reverse the matrix
    hi = np.array(wavelet.rec_hi)[::-1]
    half = size // 2
    filter_len = len(lo)
    matrix_Low_syn  = np.zeros((size, half))
    matrix_High_syn = np.zeros((size, half))

    for i in range(half):
        for j in range(filter_len):
            idx = (2 * i + j) % size
            matrix_Low_syn[idx, i]  += lo[j]
            matrix_High_syn[idx, i] += hi[j]
    return (torch.tensor(matrix_Low_syn,  dtype=torch.float32),
            torch.tensor(matrix_High_syn, dtype=torch.float32))


def idwt(LL,LH,HL,HH,matrix_Low_syn,matrix_High_syn):
        L = torch.matmul(LL, matrix_Low_syn.t()) + torch.matmul(LH, matrix_High_syn.t())
        H = torch.matmul(HL, matrix_Low_syn.t()) + torch.matmul(HH, matrix_High_syn.t())
        output = torch.matmul(matrix_Low_syn, L) + torch.matmul(matrix_High_syn, H)
        return output


if __name__ == "__main__":
    import h5py
    from PIL import Image
    import os
    import matplotlib.pyplot as plt

    with h5py.File("C:/Users/aditi/Downloads/volume_1_slice_101.h5",'r') as f:
       print(f.keys)
       img=f['image'][:, :, 3]

    img_last = torch.tensor(img).float()
    img_matrix = np.array(img)
    print(img_matrix.shape)
    H, W = img_matrix.shape
    size = max(H, W)
    matrix_Low, matrix_High = dwt_matrix(size)
    LL, LH, HL, HH = dwt(img_last,matrix_Low,matrix_High)
    print(LL.shape)

    plt.imshow(LL,cmap='gray')
    plt.show()
    plt.imshow(LH,cmap='gray')
    plt.show()
    plt.imshow(HL,cmap='gray')
    plt.show()
    plt.imshow(HH,cmap='gray')
    plt.show()

    mat_low,mat_high=idwt_matrix(size)
    img_back=idwt(LL,LH,HL,HH,mat_low,mat_high)
    plt.imshow(img_back,cmap='gray')
    plt.show()

    print(torch.allclose(img_last,img_back, atol=1e-5, rtol=1e-3))
