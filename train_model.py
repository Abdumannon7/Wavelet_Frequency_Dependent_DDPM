import argparse
import ddpm
import unet
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os
import torch.optim as optim
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import dwt_idwt_transforms as dwt_transforms
from datafilters import get_brats_dataloader



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as excep:
            print(excep)

    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # noise scheduler
    scheduler = ddpm.LinearNoiseSampler(timesteps=diffusion_config['timesteps'],
                                        beta_begin=diffusion_config['beta_begin'],
                                        beta_end=diffusion_config['beta_end']).to(device)


    # Build dataloader using BraTS function
    train_loader = get_brats_dataloader(dataset_config)

    model_LH = unet.Unet(model_config).to(device=device)
    model_LH.train()

    model_HL = unet.Unet(model_config).to(device=device)
    model_HL.train()

    model_HH = unet.Unet(model_config).to(device=device)
    model_HH.train()

    # output directories
    if not os.path.exists(train_config['output_folder']):
        os.mkdir(train_config['output_folder'])


    # train params
    num_epochs = train_config['num_epochs']
    optimizer_LH = optim.Adam(model_LH.parameters(), lr=train_config['learning_rate'])
    criterion_LH = torch.nn.L1Loss(reduction='mean')

    optimizer_HL = optim.Adam(model_HL.parameters(), lr=train_config['learning_rate'])
    criterion_HL = torch.nn.L1Loss(reduction='mean')

    optimizer_HH = optim.Adam(model_HH.parameters(), lr=train_config['learning_rate'])
    criterion_HH = torch.nn.L1Loss(reduction='mean')

    # checkpoint
    start_epoch = 0
    ckpt_path = os.path.join(train_config['output_folder'], train_config['checkpoint_file'])
    if os.path.exists(ckpt_path):
        print('Using checkpoint file')
        ckpt = torch.load(ckpt_path, map_location=device)
        model_LH.load_state_dict(ckpt['modelLH_state_dict'])
        model_HL.load_state_dict(ckpt['modelHL_state_dict'])
        model_HH.load_state_dict(ckpt['modelHH_state_dict'])
        if 'optimizerLH_state_dict' in ckpt:
            optimizer_LH.load_state_dict(ckpt['optimizerLH_state_dict'])
            optimizer_HL.load_state_dict(ckpt['optimizerHL_state_dict'])
            optimizer_HH.load_state_dict(ckpt['optimizerHH_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1

    print(f"\nTraining for {train_config['num_epochs']} epochs (starting from {start_epoch}) with batch size {dataset_config['batch_size']} on device {device}.")
    # precompute DWT matrices (constant for fixed image size)
    matrix_Low, matrix_High = dwt_transforms.dwt_matrix(dataset_config['image_size'])
    matrix_Low = matrix_Low.to(device)
    matrix_High = matrix_High.to(device)

    # epoch-level loss history for plotting
    history_LH = []
    history_HL = []
    history_HH = []

    print("\nStarting training loop")
    # training
    for epoch_idx in range(start_epoch, num_epochs):
        losses_LH = []
        losses_HL = []
        losses_HH = []

        for batch_idx, image in enumerate(train_loader):
            optimizer_LH.zero_grad()
            optimizer_HL.zero_grad()
            optimizer_HH.zero_grad()

            image = image.float().to(device)
            LL, LH, HL, HH = dwt_transforms.dwt(image, matrix_Low, matrix_High)

            LH_img = LH.float()
            HL_img = HL.float()
            HH_img = HH.float()
            x_hat = LL.float()

            # independent noise per subband
            noise_lh = torch.randn_like(LH_img).to(device)
            noise_hl = torch.randn_like(HL_img).to(device)
            noise_hh = torch.randn_like(HH_img).to(device)

            t = torch.randint(0, diffusion_config['timesteps'], (LH_img.shape[0],)).to(device)
            alpha=train_config['alpha']
            noise_LH = scheduler.loss_coeff(noise_lh, t, LH_img, x_hat)
            noisy_image_LH = scheduler.added_noise(LH_img, t, noise_lh, x_hat)
            noise_pred_LH = model_LH(torch.cat([noisy_image_LH, x_hat], dim=1), t)
            dr_LH = (noise_LH.max() - noise_LH.min()).clamp(min=1e-4).item()
            ssim_LH=1 - ssim(noise_pred_LH, noise_LH, data_range=dr_LH, size_average=True)
            loss_LH_l1 = criterion_LH(noise_pred_LH, noise_LH)
            loss_LH=(alpha*ssim_LH)+((1-alpha)*loss_LH_l1)
            losses_LH.append(loss_LH.item())
            loss_LH.backward()
            optimizer_LH.step()

            noise_HL = scheduler.loss_coeff(noise_hl, t, HL_img, x_hat)
            noisy_image_HL = scheduler.added_noise(HL_img, t, noise_hl, x_hat)
            noise_pred_HL = model_HL(torch.cat([noisy_image_HL, x_hat], dim=1), t)

            loss_HL_l1 = criterion_HL(noise_pred_HL, noise_HL)
            dr_HL = (noise_HL.max() - noise_HL.min()).clamp(min=1e-4).item()
            ssim_HL=1 - ssim(noise_pred_HL, noise_HL, data_range=dr_HL, size_average=True)
            loss_HL=(alpha*ssim_HL)+((1-alpha)*loss_HL_l1)
            losses_HL.append(loss_HL.item())
            loss_HL.backward()
            optimizer_HL.step()

            noise_HH = scheduler.loss_coeff(noise_hh, t, HH_img, x_hat)
            noisy_image_HH = scheduler.added_noise(HH_img, t, noise_hh, x_hat)
            noise_pred_HH = model_HH(torch.cat([noisy_image_HH, x_hat], dim=1), t)

            loss_HH_l1 = criterion_HH(noise_pred_HH, noise_HH)
            dr_HH = (noise_HH.max() - noise_HH.min()).clamp(min=1e-4).item()
            ssim_HH=1 - ssim(noise_pred_HH, noise_HH, data_range=dr_HH, size_average=True)
            loss_HH=(alpha*ssim_HH)+((1-alpha)*loss_HH_l1)
            losses_HH.append(loss_HH.item())
            loss_HH.backward()
            optimizer_HH.step()

        epoch_lh = np.mean(losses_LH)
        epoch_hl = np.mean(losses_HL)
        epoch_hh = np.mean(losses_HH)
        history_LH.append(epoch_lh)
        history_HL.append(epoch_hl)
        history_HH.append(epoch_hh)

        print(f'\nEpoch {epoch_idx+1}/{num_epochs} | Loss LH: {epoch_lh:.4f} | Loss HL: {epoch_hl:.4f} | Loss HH: {epoch_hh:.4f}')
        torch.save({
            'modelLH_state_dict': model_LH.state_dict(),
            'modelHL_state_dict': model_HL.state_dict(),
            'modelHH_state_dict': model_HH.state_dict(),
            'optimizerLH_state_dict': optimizer_LH.state_dict(),
            'optimizerHL_state_dict': optimizer_HL.state_dict(),
            'optimizerHH_state_dict': optimizer_HH.state_dict(),
            'epoch': epoch_idx
        }, ckpt_path)

    # save training loss curve
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, history_LH, label='LH')
    plt.plot(epochs, history_HL, label='HL')
    plt.plot(epochs, history_HH, label='HH')
    plt.xlabel('Epoch')
    plt.ylabel('MAE+SSIM Loss')
    plt.title('Wavelet DDPM Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(train_config['output_folder'], 'training_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print('Loss curve saved to', loss_plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Wavelet DDPM')
    parser.add_argument('--config_path', type=str, default='configuration.yml')
    args = parser.parse_args()
    train(args)
