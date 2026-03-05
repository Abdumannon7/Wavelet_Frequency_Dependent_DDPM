import argparse
import ddpm
import unet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import os
import yaml
from tqdm import tqdm
import numpy as np
import image_decomposition as dwt_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_normal_ll(dataset_config):
    #Load one image and return its LL subband as conditioning signal
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((dataset_config['image_size'], dataset_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.ImageFolder(root=dataset_config['train_path'], transform=transform)
    normal_class_idx = full_dataset.class_to_idx[dataset_config['target_class']]
    # find first NORMAL image
    for i, (_, label) in enumerate(full_dataset.samples):
        if label == normal_class_idx:
            img, _ = full_dataset[i]
            break
    img = img.unsqueeze(0)  # size (1, 1, H, W)
    matrix_Low, matrix_High = dwt_transforms.dwt_matrix(dataset_config['image_size'])
    LL, _, _, _ = dwt_transforms.dwt(img, matrix_Low, matrix_High)
    return LL.to(device)  # size (1, 1, H/2, W/2)


def inference(args):

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

    # load models
    model_LH = unet.Unet(model_config).to(device=device)
    model_HL = unet.Unet(model_config).to(device=device)
    model_HH = unet.Unet(model_config).to(device=device)

    ckpt_path = os.path.join(train_config['output_folder'], train_config['checkpoint_file'])
    ckpt = torch.load(ckpt_path, map_location=device)
    model_LH.load_state_dict(ckpt['modelLH_state_dict'])
    model_HL.load_state_dict(ckpt['modelHL_state_dict'])
    model_HH.load_state_dict(ckpt['modelHH_state_dict'])

    model_LH.eval()
    model_HL.eval()
    model_HH.eval()

    scheduler = ddpm.LinearNoiseSampler(timesteps=diffusion_config['timesteps'],
                                        beta_begin=diffusion_config['beta_begin'],
                                        beta_end=diffusion_config['beta_end']).to(device)

    # get real LL from a normal image
    LL = get_normal_ll(dataset_config)  # (1, 1, 32, 32)
    # expand to match number of samples
    LL = LL.expand(train_config['samples'], -1, -1, -1)  # (N, 1, 32, 32)

    with torch.no_grad():
        LH = sampling(model_LH, scheduler, train_config, model_config, diffusion_config, LL)
        HL = sampling(model_HL, scheduler, train_config, model_config, diffusion_config, LL)
        HH = sampling(model_HH, scheduler, train_config, model_config, diffusion_config, LL)

        # IDWT to reconstruct full images — use original (pre-DWT) size
        idwt_size = dataset_config['image_size']
        low_mat, high_mat = dwt_transforms.idwt_matrix(idwt_size)
        low_mat = low_mat.to(device)
        high_mat = high_mat.to(device)
        images = dwt_transforms.idwt(LL, LH, HL, HH, low_mat, high_mat)

        images = torch.clamp(images, -1, 1)
        images = (images + 1) / 2

        grid = make_grid(images, nrow=train_config['rows'])
        img_pil = transforms.ToPILImage()(grid.cpu())

        samples_dir = os.path.join(train_config['output_folder'], 'samples')
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        img_pil.save(os.path.join(samples_dir, 'x_0_final.png'))
        print('Saved sample to', os.path.join(samples_dir, 'x_0_final.png'))


def sampling(model, scheduler, train_config, model_config, diffusion_config, x_hat):

    x_t = torch.randn((train_config['samples'],
                        model_config['image_channels'],
                        model_config['image_size'],
                        model_config['image_size'])).to(device=device)

    for i in tqdm(reversed(range(diffusion_config['timesteps']))):
        noise_pred = model(torch.cat([x_t, x_hat], dim=1),
                           torch.as_tensor(i).unsqueeze(0).to(device))

        x_t, x_0_pred, _ = scheduler.sample_previous_timestep(
            x_t=x_t, noise_pred=noise_pred,
            time=torch.as_tensor(i).unsqueeze(0).to(device),
            x_hat=x_hat)

    return x_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from Wavelet DDPM')
    parser.add_argument('--config_path', type=str, default='configuration.yml')
    args = parser.parse_args()
    inference(args)
