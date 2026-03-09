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
import dwt_idwt_transforms as dwt_transforms
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from datafilters import analyze_tumor_distribution
from decode import h5_to_imgarray

from scipy.ndimage import uniform_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_ssim(img1, img2, k1=0.01, k2=0.03, win_size=7):
    # SSIM between two 2D numpy arrays in [0, 1]
    C1 = (k1 * 1.0) ** 2
    C2 = (k2 * 1.0) ** 2
    mu1 = uniform_filter(img1, size=win_size)
    mu2 = uniform_filter(img2, size=win_size)
    sigma1_sq = uniform_filter(img1 ** 2, size=win_size) - mu1 ** 2
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size) - mu2 ** 2
    sigma12 = uniform_filter(img1 * img2, size=win_size) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def get_normal_subbands(dataset_config, num_samples):
    """Load multiple images from H5 files and return all DWT subbands for conditioning and comparison"""
    
    # Analyze and filter the dataset
    sig_df, no_tumor_df = analyze_tumor_distribution(
        csv_path=dataset_config['csv_path'],
        min_slice=dataset_config.get('min_slice', 40),
        max_slice=dataset_config.get('max_slice', 124),
        tumor_threshold=dataset_config.get('tumor_threshold', 100)
    )
    
    # Select the target subset
    if dataset_config['target_class'] == 'significant_tumor':
        selected_df = sig_df
    elif dataset_config['target_class'] == 'no_tumor':
        selected_df = no_tumor_df
    else:
        raise ValueError("target_class must be 'significant_tumor' or 'no_tumor'")
    
    # Randomly select num_samples images
    indices = np.random.choice(len(selected_df), size=num_samples, replace=False)
    imgs = []
    
    for idx in indices:
        rel_path = selected_df.iloc[idx]['slice_path']
        filename = os.path.basename(rel_path)
        h5_path = os.path.join(dataset_config.get('data_root', ''), filename)
        
        # Load image using h5_to_imgarray
        img_array = h5_to_imgarray(h5_path, channel_index=dataset_config.get('channel_index', 3), normalize=True)
        
        if img_array is None:
            img_array = np.zeros((240, 240), dtype=np.uint8)
        
        # match training
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        # img_tensor = img_tensor  # old: [0, 1] range — mismatched with training
        img_tensor = (img_tensor - 0.5) / 0.5  # [-1, 1] to match datafilters.py
        imgs.append(img_tensor)
    
    imgs = torch.stack(imgs)  # (N, 1, 240, 240)
    
    # Compute DWT
    wavelet_name = dataset_config.get('wavelet', 'haar')
    matrix_Low, matrix_High = dwt_transforms.dwt_matrix(dataset_config['image_size'], wavelet_name=wavelet_name)
    LL, LH, HL, HH = dwt_transforms.dwt(imgs, matrix_Low, matrix_High)
    
    return LL.to(device), LH.to(device), HL.to(device), HH.to(device)

def normalize_subband(s):
                s = s.clone()
                s = (s - s.min()) / (s.max() - s.min() + 1e-8)
                return s

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

    wavelet_name = dataset_config.get('wavelet', 'haar')
    LL, real_LH, real_HL, real_HH = get_normal_subbands(dataset_config, train_config['samples'])

    with torch.no_grad():
        gen_LH = sampling(model_LH, scheduler, train_config, model_config, diffusion_config, LL)
        gen_HL = sampling(model_HL, scheduler, train_config, model_config, diffusion_config, LL)
        gen_HH = sampling(model_HH, scheduler, train_config, model_config, diffusion_config, LL)

        # IDWT matrices for reconstruction
        idwt_size = dataset_config['image_size']
        low_mat, high_mat = dwt_transforms.idwt_matrix(idwt_size, wavelet_name=wavelet_name)
        low_mat = low_mat.to(device)
        high_mat = high_mat.to(device)

        # generated images: real LL + generated HF
        gen_images = dwt_transforms.idwt(LL, gen_LH, gen_HL, gen_HH, low_mat, high_mat)
        gen_images = torch.clamp(gen_images, -1, 1)
        gen_images = (gen_images + 1) / 2

        samples_dir = os.path.join(train_config['output_folder'], 'samples')
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        if args.compare:
            # real images: all real subbands
            real_images = dwt_transforms.idwt(LL, real_LH, real_HL, real_HH, low_mat, high_mat)
            real_images = torch.clamp(real_images, -1, 1)
            real_images = (real_images + 1) / 2

            # interleave real and generated: [real_0, gen_0, real_1, gen_1, ...]
            N = gen_images.shape[0]
            paired = torch.stack([real_images, gen_images], dim=1).view(2 * N, *gen_images.shape[1:])
            grid = make_grid(paired, nrow=train_config['rows'])
            img_pil = transforms.ToPILImage()(grid.cpu())
            img_pil.save(os.path.join(samples_dir, 'comparison.png'))
            print('Saved comparison to', os.path.join(samples_dir, 'comparison.png'))

            for name, real_sb, gen_sb in [('LH', real_LH, gen_LH), ('HL', real_HL, gen_HL), ('HH', real_HH, gen_HH)]:
                paired_sb = torch.stack([normalize_subband(real_sb), normalize_subband(gen_sb)], dim=1)
                paired_sb = paired_sb.view(2 * N, *real_sb.shape[1:])
                grid_sb = make_grid(paired_sb, nrow=train_config['rows'])
                sb_pil = transforms.ToPILImage()(grid_sb.cpu())
                sb_pil.save(os.path.join(samples_dir, f'subbands_{name}.png'))
            # also save LL conditioning band
            grid_ll = make_grid(normalize_subband(LL), nrow=train_config['rows'])
            transforms.ToPILImage()(grid_ll.cpu()).save(os.path.join(samples_dir, 'subbands_LL.png'))
            print('Saved subband comparisons to', samples_dir)

            # compute per-image PSNR and SSIM on full reconstructed images
            psnr_vals = []
            ssim_vals = []
            for i in range(N):
                mse = torch.mean((real_images[i] - gen_images[i]) ** 2).item()
                psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
                psnr_vals.append(psnr)
                ssim_vals.append(compute_ssim(real_images[i, 0].cpu().numpy(), gen_images[i, 0].cpu().numpy()))

            print(f'\n--- Full Image Metrics ---')
            print(f'PSNR  — mean: {np.mean(psnr_vals):.2f} dB, per-image: {[f"{v:.1f}" for v in psnr_vals]}')
            print(f'SSIM  — mean: {np.mean(ssim_vals):.4f}, per-image: {[f"{v:.3f}" for v in ssim_vals]}')

            # compute per-subband PSNR, SSIM, and MAE on raw coefficient matrices
            print(f'\n--- Subband Metrics (raw DWT coefficients) ---')
            for name, real_sb, gen_sb in [('LH', real_LH, gen_LH), ('HL', real_HL, gen_HL), ('HH', real_HH, gen_HH)]:
                sb_psnr = []
                sb_ssim = []
                sb_mae = []
                for i in range(N):
                    r = real_sb[i, 0].cpu().numpy()
                    g = gen_sb[i, 0].cpu().numpy()
                    # normalize both to [0,1] using shared range for fair comparison
                    lo = min(r.min(), g.min())
                    hi = max(r.max(), g.max())
                    r_norm = (r - lo) / (hi - lo + 1e-8)
                    g_norm = (g - lo) / (hi - lo + 1e-8)
                    mse = np.mean((r_norm - g_norm) ** 2)
                    sb_psnr.append(10 * np.log10(1.0 / max(mse, 1e-10)))
                    sb_ssim.append(compute_ssim(r_norm, g_norm))
                    sb_mae.append(np.mean(np.abs(r - g)))
                print(f'{name}  PSNR: {np.mean(sb_psnr):.2f} dB | SSIM: {np.mean(sb_ssim):.4f} | MAE: {np.mean(sb_mae):.4f}')

            # save amplified difference maps to show where generated HF deviates from real
            fig, axes = plt.subplots(3, N, figsize=(2 * N, 6))
            if N == 1:
                axes = axes.reshape(3, 1)
            for row, (name, real_sb, gen_sb) in enumerate([('LH', real_LH, gen_LH), ('HL', real_HL, gen_HL), ('HH', real_HH, gen_HH)]):
                for col in range(N):
                    diff = (gen_sb[col, 0] - real_sb[col, 0]).cpu().numpy()
                    im = axes[row, col].imshow(diff, cmap='RdBu_r', vmin=-diff.std()*3, vmax=diff.std()*3)
                    axes[row, col].axis('off')
                    if col == 0:
                        axes[row, col].set_ylabel(name, fontsize=12)
            fig.suptitle('Generated − Real HF subbands (blue=lower, red=higher)', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(samples_dir, 'subband_diff_map.png'), dpi=150)
            plt.close(fig)

            # print how much the generated subbands actually differ from real
            print(f'\n--- Subband Deviation (generated vs real) ---')
            for name, real_sb, gen_sb in [('LH', real_LH, gen_LH), ('HL', real_HL, gen_HL), ('HH', real_HH, gen_HH)]:
                diff = (gen_sb - real_sb).cpu().numpy()
                real_np = real_sb.cpu().numpy()
                # relative error: how large is the difference compared to the real signal energy
                rel_err = np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(real_np ** 2)) + 1e-8) * 100
                print(f'{name}  mean_diff: {np.mean(diff):+.4f} | std_diff: {np.std(diff):.4f} | '
                      f'real_std: {np.std(real_np):.4f} | relative_error: {rel_err:.1f}%')
        else:
            grid = make_grid(gen_images, nrow=train_config['rows'])
            img_pil = transforms.ToPILImage()(grid.cpu())
            img_pil.save(os.path.join(samples_dir, 'x_0_final.png'))
            print('Saved sample to', os.path.join(samples_dir, 'x_0_final.png'))




def sampling(model, scheduler, train_config, model_config, diffusion_config, x_hat):

    # x_t = torch.randn((train_config['samples'],
    #                     model_config['image_channels'],
    #                     model_config['image_size'],
    #                     model_config['image_size'])).to(device=device)

    # Algorithm 2: x_T ~ N(sqrt(alpha_bar_T) * x_hat, delta_T * I)
    T = diffusion_config['timesteps'] - 1
    sqrt_alpha_bar_T = scheduler.alpha_cumulative_sqrt[T]
    delta_T = scheduler.deltas[T]
    z = torch.randn((train_config['samples'],
                      model_config['image_channels'],
                      model_config['image_size'],
                      model_config['image_size'])).to(device=device)
    x_t = sqrt_alpha_bar_T * x_hat + torch.sqrt(delta_T) * z

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
    parser.add_argument('--compare', default=True, action='store_true', help='Compare generated vs real images with metrics')
    args = parser.parse_args()
    inference(args)
