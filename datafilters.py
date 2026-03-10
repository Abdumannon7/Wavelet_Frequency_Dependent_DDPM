import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from decode import h5_to_imgarray
import numpy as np
import os


def analyze_tumor_distribution(csv_path, min_slice=40, max_slice=124, tumor_threshold=100, downsample_bins=True):
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
    if downsample_bins:
        valid_slices = list(range(min_slice, max_slice + 1, 5))
        mid_brain_df = df[df['slice'].isin(valid_slices)].copy()
    else:
        mid_brain_df = df[(df['slice'] >= min_slice) & (df['slice'] <= max_slice)].copy()

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



def balance_by_slice_bins(df, min_slice=40, max_slice=130, bin_size=10, random_seed=42):
    """
    Balances the dataset by capping the number of samples in each slice bin
    to the minimum count between 'tumor' and 'no_tumor' in that bin.
    """
    print("\n--- Starting Stratified Downsampling ---")
    
    # 1. Define bins and labels
    bins = list(range(min_slice, max_slice + 1, bin_size))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    
    # 2. Create the bin column
    df = df.copy()
    df['slice_bin'] = pd.cut(df['slice'], bins=bins, labels=labels, right=False)
    
    # 3. Find the limiting factor (minimum count) for each bin
    # Unstack pivots the tumor_class into columns so we can easily find the min row-wise
    counts = df.groupby(['slice_bin', 'tumor_class'], observed=False).size().unstack(fill_value=0)
    min_counts = counts.min(axis=1)
    
    balanced_dfs = []
    
    # 4. Sample exactly the minimum amount from each class per bin
    for bin_label in labels:
        limit = min_counts[bin_label]
        
        if limit == 0:
            print(f"Skipping bin {bin_label} (Limit is 0)")
            continue
            
        print(f"Bin {bin_label}: Capping at {limit} samples per class.")
        
        for tumor_class in df['tumor_class'].unique():
            # Filter for the specific bin and class
            subset = df[(df['slice_bin'] == bin_label) & (df['tumor_class'] == tumor_class)]
            
            # Randomly sample 'limit' rows
            sampled_subset = subset.sample(n=limit, random_state=random_seed)
            balanced_dfs.append(sampled_subset)
            
    # 5. Combine the perfectly balanced subsets back together
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    
    print("--- Downsampling Complete ---\n")
    return balanced_df


def create_train_val_test_split(csv_path, output_csv_path, 
                                no_tumor_threshold=100, significant_tumor_threshold=1400,
                                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                min_slice=40, max_slice=124, random_seed=42):
    
    np.random.seed(random_seed)
    
    print(f"Filtering with thresholds:")
    print(f"  - No Tumor: < {no_tumor_threshold} pixels")
    print(f"  - Significant Tumor: >= {significant_tumor_threshold} pixels\n")
    
    # Use analyze_tumor_distribution to get dataframes
    # (Assuming analyze_tumor_distribution is defined elsewhere in your code)
    sig_tumor_at_threshold, no_tumor_df = analyze_tumor_distribution(
        csv_path=csv_path, min_slice=min_slice, max_slice=max_slice, tumor_threshold=no_tumor_threshold
    )
    
    significant_tumor_df, _ = analyze_tumor_distribution(
        csv_path=csv_path, min_slice=min_slice, max_slice=max_slice, tumor_threshold=significant_tumor_threshold
    )
    
    print(f"Dataset sizes after filtering (Before Balancing):")
    print(f"  - No Tumor slices: {len(no_tumor_df)}")
    print(f"  - Significant Tumor slices: {len(significant_tumor_df)}")
    
    # 1. Add tumor class labels FIRST
    no_tumor_df['tumor_class'] = 'no_tumor'
    significant_tumor_df['tumor_class'] = 'significant_tumor'
    
    # 2. Combine the dataframes BEFORE splitting
    combined_df = pd.concat([no_tumor_df, significant_tumor_df], ignore_index=True)
    
    # 3. BALANCE THE DATASET (Using our new function)
    # Using max_slice=130 to safely encompass your 124 limit within the 120-130 bin
    balanced_df = balance_by_slice_bins(combined_df, min_slice=40, max_slice=130, bin_size=10, random_seed=random_seed)
    
    # 4. Helper function to create splits (Modified to work on the balanced df)
    def assign_splits(dataframe, train_ratio, val_ratio, test_ratio, seed):
        """Assign train/val/test labels to dataframe."""
        n = len(dataframe)
        indices = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = np.array(['train'] * n)
        splits[indices[train_end:val_end]] = 'val'
        splits[indices[val_end:]] = 'test'
        
        # Avoid SettingWithCopyWarning by returning a copy
        df_copy = dataframe.copy()
        df_copy['split'] = splits
        return df_copy

    # 5. Apply splits per class to ensure exactly equal representation in train/val/test
    final_dfs = []
    for tumor_class in ['no_tumor', 'significant_tumor']:
        class_df = balanced_df[balanced_df['tumor_class'] == tumor_class]
        split_class_df = assign_splits(class_df, train_ratio, val_ratio, test_ratio, random_seed)
        final_dfs.append(split_class_df)
        
    final_combined_df = pd.concat(final_dfs, ignore_index=True)
    
    # Print split statistics
    print(f"Train/Val/Test splits (After Balancing):")
    for tumor_class in ['no_tumor', 'significant_tumor']:
        class_df = final_combined_df[final_combined_df['tumor_class'] == tumor_class]
        print(f"\n  {tumor_class}:")
        for split in ['train', 'val', 'test']:
            count = len(class_df[class_df['split'] == split])
            pct = (count / len(class_df) * 100) if len(class_df) > 0 else 0
            print(f"    - {split}: {count} ({pct:.1f}%)")
            
    # Optionally drop the 'slice_bin' column if you don't need it for training
    final_combined_df = final_combined_df.drop(columns=['slice_bin'], errors='ignore')

    # Save to CSV
    final_combined_df.to_csv(output_csv_path, index=False)
    print(f"\nPerfectly balanced dataset saved to: {output_csv_path}")
    print(f"  Total samples: {len(final_combined_df)}")
    
    return final_combined_df


def create_train_val_test_split_volume_level(csv_path, output_csv_path, 
                                             no_tumor_threshold=100, significant_tumor_threshold=1400,
                                             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                             min_slice=40, max_slice=124, random_seed=42):
    
    print(f"Filtering with thresholds: No Tumor < {no_tumor_threshold}, Significant Tumor >= {significant_tumor_threshold}\n")
    
    # 1. Load and filter the data (assuming analyze_tumor_distribution is available)
    _, no_tumor_df = analyze_tumor_distribution(
        csv_path=csv_path, min_slice=min_slice, max_slice=max_slice, tumor_threshold=no_tumor_threshold
    )
    significant_tumor_df, _ = analyze_tumor_distribution(
        csv_path=csv_path, min_slice=min_slice, max_slice=max_slice, tumor_threshold=significant_tumor_threshold
    )
    
    no_tumor_df['tumor_class'] = 'no_tumor'
    significant_tumor_df['tumor_class'] = 'significant_tumor'
    
    # Combine everything into one master dataframe
    combined_df = pd.concat([no_tumor_df, significant_tumor_df], ignore_index=True)
    
    # ---------------------------------------------------------
    # 2. PATIENT-LEVEL (VOLUME) SPLITTING
    # ---------------------------------------------------------
    np.random.seed(random_seed)
    unique_volumes = combined_df['volume'].unique()
    np.random.shuffle(unique_volumes)
    
    n_vols = len(unique_volumes)
    train_end = int(n_vols * train_ratio)
    val_end = train_end + int(n_vols * val_ratio)
    
    train_vols = set(unique_volumes[:train_end])
    val_vols = set(unique_volumes[train_end:val_end])
    test_vols = set(unique_volumes[val_end:])
    
    # Map the volumes to their respective splits
    def assign_volume_split(vol):
        if vol in train_vols: return 'train'
        if vol in val_vols: return 'val'
        return 'test'
        
    combined_df['split'] = combined_df['volume'].apply(assign_volume_split)
    
    print(f"Volume-Level Split Complete:")
    print(f"  - Train Volumes: {len(train_vols)}")
    print(f"  - Val Volumes: {len(val_vols)}")
    print(f"  - Test Volumes: {len(test_vols)}\n")

    # ---------------------------------------------------------
    # 3. INDEPENDENT BALANCING PER SPLIT
    # ---------------------------------------------------------
    # We balance train, val, and test separately to ensure perfect shape distribution 
    # within each phase, while maintaining strict patient isolation.
    
    final_dfs = []
    for split_name in ['train', 'val', 'test']:
        print(f"--- Balancing {split_name.upper()} Set ---")
        split_df = combined_df[combined_df['split'] == split_name].copy()
        
        # Balance using max_slice=130 to encompass the 124 limit
        balanced_split_df = balance_by_slice_bins(
            split_df, min_slice=40, max_slice=130, bin_size=10, random_seed=random_seed
        )
        final_dfs.append(balanced_split_df)
        
    # Combine the balanced, strictly-isolated sets
    final_combined_df = pd.concat(final_dfs, ignore_index=True)
    
    # Clean up and save
    final_combined_df = final_combined_df.drop(columns=['slice_bin'], errors='ignore')
    final_combined_df.to_csv(output_csv_path, index=False)
    
    print(f"\nPerfectly balanced, strictly isolated dataset saved to: {output_csv_path}")
    print(f"  Total samples: {len(final_combined_df)}")
    
    return final_combined_df


if __name__ == '__main__':
    # Example usage
    csv_path = 'F:/BraTS2020/BraTS20 Training Metadata.csv'  # Update with your actual CSV path
    output_csv_path = 'brats_train_val_test_split.csv'
    
    # Create train/val/test splits with custom thresholds
    combined_df = create_train_val_test_split_volume_level(
        csv_path=csv_path,
        output_csv_path=output_csv_path,
        no_tumor_threshold=100,
        significant_tumor_threshold=1400,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )