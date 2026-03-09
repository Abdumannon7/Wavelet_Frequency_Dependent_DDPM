import numpy as np
import torch
import pywt

def dwt_matrix(size, wavelet_name='haar'):
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


