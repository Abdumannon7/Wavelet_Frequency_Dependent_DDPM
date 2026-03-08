import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from decode import h5_to_imgarray
import numpy as np
import os


def analyze_tumor_distribution(csv_path, min_slice=40, max_slice=124, tumor_threshold=100):
    """
    Reads BraTS metadata and filters slices based on their axial position and tumor pixel count.
    
    Args:
        csv_path (str): Path to the BraTS20 Training Metadata.csv
        min_slice (int): Lower bound for slice index (inclusive). Defaults to 40.
        max_slice (int): Upper bound for slice index (inclusive). Defaults to 124.
                         (Assumes 155 total slices: 154 - 30 = 124)
        tumor_threshold (int): Minimum total tumor pixels to be considered "Significant".
    """
    print(f"Loading metadata from: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    initial_count = len(df)
    print(f"Total slices in dataset: {initial_count}")
    
    # --- Filter 1: Exclude extreme top and bottom slices ---
    # Keep only the slices in the "mid-brain" region
    valid_slices = list(range(min_slice, max_slice + 1, 5))
    mid_brain_df = df[df['slice'].isin(valid_slices)].copy()
    mid_brain_count = len(mid_brain_df)
    print(f"\nApplied Slice Filter (Keep slices {min_slice} to {max_slice}):")
    print(f"  -> Slices retained: {mid_brain_count} (Excluded {initial_count - mid_brain_count} extreme slices)")
    
    # --- Filter 2: Tumor vs. No/Little Tumor ---
    # Calculate total tumor pixels by summing all three label channels
    mid_brain_df['total_tumor_pixels'] = (
        mid_brain_df['label0_pxl_cnt'] + 
        mid_brain_df['label1_pxl_cnt'] + 
        mid_brain_df['label2_pxl_cnt']
    )
    
    # Split the dataframe based on the threshold
    significant_tumor_df = mid_brain_df[mid_brain_df['total_tumor_pixels'] >= tumor_threshold]
    no_tumor_df = mid_brain_df[mid_brain_df['total_tumor_pixels'] < tumor_threshold]
    
    sig_count = len(significant_tumor_df)
    no_count = len(no_tumor_df)
    
    print(f"\nApplied Tumor Threshold (>= {tumor_threshold} total tumor pixels):")
    print(f"  -> Significant Tumor slices: {sig_count}")
    print(f"  -> No/Little Tumor slices:   {no_count}")
    
    # Calculate class balance ratio
    if no_count > 0:
        ratio = sig_count / no_count
        print(f"\nClass Balance Ratio (Tumor:No Tumor) in filtered data: 1 : {1/ratio:.2f}")
    
    # Return the filtered dataframes so you can use them directly in your PyTorch Datasets
    return significant_tumor_df, no_tumor_df



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
        filename = os.path.basename(rel_path)
        h5_path = os.path.join(self.data_root, filename) if self.data_root else rel_path
        
        # Fetch the normalized 0-255 uint8 numpy array
        img_array = h5_to_imgarray(h5_path, channel_index=self.channel_index, normalize=True)
        
        # Fallback for corrupted files
        if img_array is None:
            img_array = np.zeros((240, 240), dtype=np.uint8)
            
        # Convert to PyTorch Tensor. 
       
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        img_tensor=(img_tensor-0.5)/0.5  #normalised the image 
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