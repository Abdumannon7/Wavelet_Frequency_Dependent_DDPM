# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch
from torch.autograd import Function

class DWTFunction_2D(Function):
    # @staticmethod
    def forward(input, matrix_Low, matrix_High):
        L = torch.matmul(matrix_Low, input)
        H = torch.matmul(matrix_High, input)
        LL = torch.matmul(L, matrix_Low.t())
        LH = torch.matmul(L, matrix_High.t())
        HL = torch.matmul(H, matrix_Low.t())
        HH = torch.matmul(H, matrix_High.t())
        nonzero_mask_Low = (matrix_Low != 0).float()
        # nonzero_mask_Low_1 = (         != 0).float()
        nonzero_mask_High = (matrix_High != 0).float()
        # nonzero_mask_High_1 = (matrix_High_1 != 0).float()
        # ctx.save_for_backward(input, matrix_Low, matrix_High, L, H, nonzero_mask_Low, nonzero_mask_High)

        return LL, LH, HL, HH
    # def forward(ctx, input, matrix_Low, matrix_High):
    #     L = torch.matmul(matrix_Low, input)
    #     H = torch.matmul(matrix_High, input)
    #     LL = torch.matmul(L, matrix_Low.t())
    #     LH = torch.matmul(L, matrix_High.t())
    #     HL = torch.matmul(H, matrix_Low.t())
    #     HH = torch.matmul(H, matrix_High.t())
    #     nonzero_mask_Low = (matrix_Low != 0).float()
    #     # nonzero_mask_Low_1 = (         != 0).float()
    #     nonzero_mask_High = (matrix_High != 0).float()
    #     # nonzero_mask_High_1 = (matrix_High_1 != 0).float()
    #     ctx.save_for_backward(input, matrix_Low, matrix_High, L, H, nonzero_mask_Low, nonzero_mask_High)

    #     return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        input, matrix_Low, matrix_High, L, H, nonzero_mask_Low, nonzero_mask_High = ctx.saved_tensors
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low), torch.matmul(grad_LH, matrix_High))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low), torch.matmul(grad_HH, matrix_High))
        grad_input = torch.add(torch.matmul(matrix_Low.t(), grad_L), torch.matmul(matrix_High.t(), grad_H))

        grad_matrix_Low = torch.matmul(grad_L, input.transpose(-2, -1))
        # grad_matrix_Low_1 = torch.matmul(L.transpose(-2, -1), grad_LL) + torch.matmul(H.transpose(-2, -1), grad_HL)
        grad_matrix_High = torch.matmul(grad_H, input.transpose(-2, -1))
        # grad_matrix_High_1 = torch.matmul(L.transpose(-2, -1), grad_LH) + torch.matmul(H.transpose(-2, -1), grad_HH)

        grad_matrix_Low = grad_matrix_Low * nonzero_mask_Low
        grad_matrix_High = grad_matrix_High * nonzero_mask_High

        return grad_input, grad_matrix_Low, grad_matrix_High

class IDWTFunction_2D(Function):
    """
    2D IDWT matching DWTFunction_2D structure.
    Synthesis matrices from get_matrix are already sized for upsampling.
    """
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH, matrix_Low_synth, matrix_High_synth):
        ctx.save_for_backward(matrix_Low_synth, matrix_High_synth)

        # Upsample width: combine LL+LH and HL+HH
        # matrix_synth is (subband_size, output_size) for upsampling
        L = torch.add(torch.matmul(input_LL, matrix_Low_synth),
                      torch.matmul(input_LH, matrix_High_synth))
        H = torch.add(torch.matmul(input_HL, matrix_Low_synth),
                      torch.matmul(input_HH, matrix_High_synth))

        # Upsample height: combine L+H
        # Need to transpose matrix to multiply on left
        output = torch.add(torch.matmul(matrix_Low_synth.t(), L),
                          torch.matmul(matrix_High_synth.t(), H))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_synth, matrix_High_synth = ctx.saved_tensors

        # Gradient w.r.t. L and H
        grad_L = torch.matmul(matrix_Low_synth, grad_output)
        grad_H = torch.matmul(matrix_High_synth, grad_output)

        # Gradient w.r.t. subbands
        grad_LL = torch.matmul(grad_L, matrix_Low_synth.t())
        grad_LH = torch.matmul(grad_L, matrix_High_synth.t())
        grad_HL = torch.matmul(grad_H, matrix_Low_synth.t())
        grad_HH = torch.matmul(grad_H, matrix_High_synth.t())

        return grad_LL, grad_LH, grad_HL, grad_HH, None, None