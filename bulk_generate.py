import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import unet
import ddpm
import dwt_idwt_transforms as dwt_transforms
from decode import h5_to_imgarray

# Reuse the sampling function from your existing script
from sample_model import sampling 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bulk_generate(args):
    # 1. Load Configurations
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # 2. Setup Output Directories
    output_dir = os.path.join(train_config['output_folder'], 'synthetic_data')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_output_path = os.path.join(train_config['output_folder'], 'synthetic_metadata.csv')

    # 3. Load the 3 DDPM Models
    print("Loading DDPM Models...")
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

    scheduler = ddpm.LinearNoiseSampler(
        timesteps=diffusion_config['timesteps'],
        beta_begin=diffusion_config['beta_begin'],
        beta_end=diffusion_config['beta_end']
    ).to(device)

    # 4. Read and Filter the Source CSV
    # We strictly use the 'train' split to avoid data leakage
    print("Reading source dataset...")
    classifier_config = config.get('classifier', {})
    source_csv = classifier_config.get('csv_path', 'brats_train_val_test_split.csv')
    df = pd.read_csv(source_csv)
    
    # Filter for training data and the specific class you want to augment
    target_class = dataset_config.get('target_class', 'significant_tumor')
    target_df = df[(df['split'] == 'train') & (df['tumor_class'] == target_class)].copy()
    
    print(f"Found {len(target_df)} source images for class '{target_class}' in the training split.")

    if args.max_samples and len(target_df) > args.max_samples:
        print(f"Found {len(target_df)} source images. Downsampling to {args.max_samples}...")
        target_df = target_df.sample(n=args.max_samples, random_state=42).copy()
    else:
        print(f"Found {len(target_df)} source images for class '{target_class}' in the training split.")

    # 5. DWT / IDWT Matrices setup
    wavelet_name = dataset_config.get('wavelet', 'haar')
    img_size = dataset_config['image_size']
    matrix_Low, matrix_High = dwt_transforms.dwt_matrix(img_size, wavelet_name=wavelet_name)
    low_mat, high_mat = dwt_transforms.idwt_matrix(img_size, wavelet_name=wavelet_name)
    low_mat, high_mat = low_mat.to(device), high_mat.to(device)

    # 6. Batched Generation Loop
    batch_size = train_config.get('samples', 16)
    synthetic_records = []
    
    # Process in batches because DDPM inference is slow
    for start_idx in tqdm(range(0, len(target_df), batch_size), desc="Bulk Generating"):
        end_idx = min(start_idx + batch_size, len(target_df))
        batch_df = target_df.iloc[start_idx:end_idx]
        
        # Prepare batch tensors
        imgs = []
        filenames = []
        
        for _, row in batch_df.iterrows():
            rel_path = row['slice_path']
            h5_path = os.path.join(dataset_config.get('data_root', ''), rel_path)
            
            img_array = h5_to_imgarray(h5_path, channel_index=dataset_config.get('channel_index', 3), normalize=True)
            if img_array is None:
                img_array = np.zeros((240, 240), dtype=np.uint8)
                
            img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5  # Scale to [-1, 1]
            imgs.append(img_tensor)
            
            # Create a clean filename for the new synthetic array
            base_name = os.path.basename(rel_path).replace('.h5', '')
            filenames.append(f"synth_{base_name}.npy")
            
        imgs = torch.stack(imgs) # Shape: (B, 1, H, W)
        
        # Apply DWT to get the Real LL band for conditioning
        LL, _, _, _ = dwt_transforms.dwt(imgs, matrix_Low, matrix_High)
        LL = LL.to(device)

        # Generate synthetic High Frequency bands
        with torch.no_grad():
            gen_LH = sampling(model_LH, scheduler, train_config, model_config, diffusion_config, LL)
            gen_HL = sampling(model_HL, scheduler, train_config, model_config, diffusion_config, LL)
            gen_HH = sampling(model_HH, scheduler, train_config, model_config, diffusion_config, LL)

            # Reconstruct the image using IDWT
            gen_images = dwt_transforms.idwt(LL, gen_LH, gen_HL, gen_HH, low_mat, high_mat)
            gen_images = torch.clamp(gen_images, -1, 1)
            
            # Scale back to [0, 1] then [0, 255] for saving as standard image arrays
            gen_images = (gen_images + 1) / 2.0
            gen_images_np = (gen_images.cpu().numpy() * 255).astype(np.uint8)

        # Save individual .npy files and record metadata
        for i, filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            
            # Squeeze out the channel dim so it's a standard (240, 240) 2D array
            single_img_array = gen_images_np[i, 0] 
            np.save(save_path, single_img_array)
            
            # Record for the new CSV. We flag it as 'synthetic' just in case you need to filter later.
            synthetic_records.append({
                'slice_path': os.path.join('synthetic_data', filename), # Relative path
                'target': batch_df.iloc[i]['target'], # Keep original numerical target
                'tumor_class': target_class,
                'split': 'train', 
                'is_synthetic': True 
            })

    # 7. Save the new CSV
    synth_df = pd.DataFrame(synthetic_records)
    synth_df.to_csv(csv_output_path, index=False)
    
    print("\n" + "="*50)
    print(f"Generation Complete! Created {len(synth_df)} synthetic arrays.")
    print(f"Saved arrays to: {output_dir}")
    print(f"Saved metadata to: {csv_output_path}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bulk Generate Synthetic Data')
    parser.add_argument('--config_path', type=str, default='configuration.yml')

    parser.add_argument('--max_samples', type=int, default=500, help='Maximum number of synthetic images to generate')

    args = parser.parse_args()
    
    bulk_generate(args)