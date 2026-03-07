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
import image_decomposition as dwt_transforms
from datafilters import analyze_tumor_distribution
from decode import h5_to_imgarray


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BraTSBasicDataset(Dataset):
    def __init__(self, dataframe, data_root='', channel_index=3):
        """
        Args:
            dataframe (pd.DataFrame): Filtered dataframe containing slice paths.
            data_root (str): Base path to append to the relative paths in the CSV.
            channel_index (int): Modality index (3 = FLAIR).
        """
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.channel_index = channel_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.loc[idx, 'slice_path']
        h5_path = os.path.join(self.data_root, rel_path) if self.data_root else rel_path
        
        # Fetch the normalized 0-255 uint8 numpy array
        img_array = h5_to_imgarray(h5_path, channel_index=self.channel_index, normalize=True)
        
        # Fallback for corrupted files
        if img_array is None:
            img_array = np.zeros((240, 240), dtype=np.uint8)
            
        # Convert to PyTorch Tensor. 
        # We scale / 255.0 to give your friends a nice [0.0, 1.0] float32 tensor to work with.
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        
        return img_tensor


def get_brats_dataloader(config):
    """
    Builds the dataloader based on the configuration dictionary.
    """
    print("Filtering metadata...")
    sig_df, no_tumor_df = analyze_tumor_distribution(
        csv_path=config['csv_path'],
        min_slice=config.get('min_slice', 40),
        max_slice=config.get('max_slice', 124),
        tumor_threshold=config.get('tumor_threshold', 100)
    )

    # Select the target subset
    if config['target_class'] == 'significant_tumor':
        selected_df = sig_df
    elif config['target_class'] == 'no_tumor':
        selected_df = no_tumor_df
    else:
        raise ValueError("config['target_class'] must be 'significant_tumor' or 'no_tumor'")
        
    print(f"--> Building DataLoader with {len(selected_df)} {config['target_class']} slices.")

    # Instantiate Dataset
    dataset = BraTSBasicDataset(
        dataframe=selected_df, 
        data_root=config.get('data_root', ''),
        channel_index=config.get('channel_index', 3) # Default to FLAIR
    )
    
    # Instantiate DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=config.get('batch_size'), 
        shuffle=config.get('shuffle_bool', True), 
        num_workers=config.get('num_workers'),
        pin_memory=False
    )
    
    return dataloader


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

    # # dataset — NORMAL class only
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((dataset_config['image_size'], dataset_config['image_size'])),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    # full_dataset = datasets.ImageFolder(root=dataset_config['train_path'], transform=transform)
    # normal_class_idx = full_dataset.class_to_idx[dataset_config['target_class']]
    # normal_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == normal_class_idx]
    # train_dataset = torch.utils.data.Subset(full_dataset, normal_indices)

    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=train_config['batch_size'],
    #                           shuffle=train_config['shuffle_bool'],
    #                           num_workers=dataset_config['num_workers'])

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


    # checkpoint
    ckpt_path = os.path.join(train_config['output_folder'], train_config['checkpoint_file'])
    if os.path.exists(ckpt_path):
        print('Using checkpoint file')
        ckpt = torch.load(ckpt_path, map_location=device)
        model_LH.load_state_dict(ckpt['modelLH_state_dict'])
        model_HL.load_state_dict(ckpt['modelHL_state_dict'])
        model_HH.load_state_dict(ckpt['modelHH_state_dict'])

    # train params
    print(f"\nTraining for {train_config['num_epochs']} epochs with batch size {dataset_config['batch_size']} on device {device}.")

    num_epochs = train_config['num_epochs']
    optimizer_LH = optim.Adam(model_LH.parameters(), lr=train_config['learning_rate'])
    criterion_LH = torch.nn.MSELoss()

    optimizer_HL = optim.Adam(model_HL.parameters(), lr=train_config['learning_rate'])
    criterion_HL = torch.nn.MSELoss()

    optimizer_HH = optim.Adam(model_HH.parameters(), lr=train_config['learning_rate'])
    criterion_HH = torch.nn.MSELoss()

    # precompute DWT matrices (constant for fixed image size)
    matrix_Low, matrix_High = dwt_transforms.dwt_matrix(dataset_config['image_size'])
    matrix_Low = matrix_Low.to(device)
    matrix_High = matrix_High.to(device)

    # epoch-level loss history for plotting
    history_LH = []
    history_HL = []
    history_HH = []

    print("\nStarting training loop...")
    # training
    for epoch_idx in range(num_epochs):
        losses_LH = []
        losses_HL = []
        losses_HH = []

        for batch_idx, image in enumerate(tqdm(train_loader, desc=f'Epoch {epoch_idx+1}/{num_epochs}')):
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

            noise_LH = scheduler.loss_coeff(noise_lh, t, LH_img, x_hat)
            noisy_image_LH = scheduler.added_noise(LH_img, t, noise_lh, x_hat)
            noise_pred_LH = model_LH(torch.cat([noisy_image_LH, x_hat], dim=1), t)

            loss_LH = criterion_LH(noise_pred_LH, noise_LH)
            losses_LH.append(loss_LH.item())
            loss_LH.backward()
            optimizer_LH.step()

            noise_HL = scheduler.loss_coeff(noise_hl, t, HL_img, x_hat)
            noisy_image_HL = scheduler.added_noise(HL_img, t, noise_hl, x_hat)
            noise_pred_HL = model_HL(torch.cat([noisy_image_HL, x_hat], dim=1), t)

            loss_HL = criterion_HL(noise_pred_HL, noise_HL)
            losses_HL.append(loss_HL.item())
            loss_HL.backward()
            optimizer_HL.step()

            noise_HH = scheduler.loss_coeff(noise_hh, t, HH_img, x_hat)
            noisy_image_HH = scheduler.added_noise(HH_img, t, noise_hh, x_hat)
            noise_pred_HH = model_HH(torch.cat([noisy_image_HH, x_hat], dim=1), t)

            loss_HH = criterion_HH(noise_pred_HH, noise_HH)
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
            'modelHH_state_dict': model_HH.state_dict()
        }, ckpt_path)

    # save training loss curve
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, history_LH, label='LH')
    plt.plot(epochs, history_HL, label='HL')
    plt.plot(epochs, history_HH, label='HH')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
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
