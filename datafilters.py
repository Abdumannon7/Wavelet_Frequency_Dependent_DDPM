import pandas as pd


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