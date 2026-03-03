"""
Convert HDF5 brain tumor MRI data to JPG images for visualization
"""
import h5py
import numpy as np
from PIL import Image
import os


def h5_to_jpg(h5_path, output_jpg_path, channel_index=0, normalize=True):
    """
    Convert a single channel from an HDF5 file to a JPG image.
    
    Args:
        h5_path (str): Path to the .h5 file
        output_jpg_path (str): Path where the JPG image will be saved
        channel_index (int): Which channel to extract (0-3 for the 4 channels in image data)
        normalize (bool): Whether to normalize the image to 0-255 range
    """
    try:
        # Read the HDF5 file
        with h5py.File(h5_path, 'r') as f:
            # Get the image data - shape is (240, 240, 4)
            image_data = f['image'][:]
        
        # Extract the specified channel
        if image_data.shape[2] <= channel_index:
            raise ValueError(f"Channel index {channel_index} is out of range. Image has {image_data.shape[2]} channels.")
        
        channel_data = image_data[:, :, channel_index]
        
        # Normalize to 0-255 range
        if normalize:
            # Handle case where all values are the same
            if np.max(channel_data) == np.min(channel_data):
                img_array = np.zeros_like(channel_data, dtype=np.uint8)
            else:
                img_array = ((channel_data - np.min(channel_data)) / 
                            (np.max(channel_data) - np.min(channel_data)) * 255).astype(np.uint8)
        else:
            img_array = np.clip(channel_data, 0, 255).astype(np.uint8)
        
        # Create PIL Image and save as JPG
        img = Image.fromarray(img_array, mode='L')
        os.makedirs(os.path.dirname(output_jpg_path), exist_ok=True)
        img.save(output_jpg_path, 'JPEG', quality=95)
        
        print(f"Successfully converted: {h5_path}")
        print(f"  Saved to: {output_jpg_path}")
        print(f"  Channel: {channel_index}, Data range: [{np.min(channel_data):.2f}, {np.max(channel_data):.2f}]")
        
        return True
        
    except Exception as e:
        print(f"Error converting {h5_path}: {str(e)}")
        return False


def h5_to_jpg_all_channels(h5_path, output_dir, normalize=True):
    """
    Convert all 4 channels from an HDF5 file to separate JPG images.
    
    Args:
        h5_path (str): Path to the .h5 file
        output_dir (str): Directory where JPG images will be saved
        normalize (bool): Whether to normalize images to 0-255 range
    """
    try:
        # Read the HDF5 file
        with h5py.File(h5_path, 'r') as f:
            image_data = f['image'][:]
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        
        # Process each channel
        for channel_idx in range(image_data.shape[2]):
            channel_data = image_data[:, :, channel_idx]
            
            # Normalize to 0-255 range
            if normalize:
                if np.max(channel_data) == np.min(channel_data):
                    img_array = np.zeros_like(channel_data, dtype=np.uint8)
                else:
                    img_array = ((channel_data - np.min(channel_data)) / 
                                (np.max(channel_data) - np.min(channel_data)) * 255).astype(np.uint8)
            else:
                img_array = np.clip(channel_data, 0, 255).astype(np.uint8)
            
            # Save as JPG
            output_path = os.path.join(output_dir, f"{base_name}_channel_{channel_idx}.jpg")
            img = Image.fromarray(img_array, mode='L')
            img.save(output_path, 'JPEG', quality=95)
            print(f"  Channel {channel_idx} -> {output_path}")
        
        print(f"All channels converted from: {h5_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {h5_path}: {str(e)}")
        return False




def get_mask_channel(h5_path, channel_index=0):
    """Return a 2-D array corresponding to one channel of the mask.

    The mask dataset has shape (240,240,3); channel_index must be 0-2.
    """
    with h5py.File(h5_path, 'r') as f:
        mask = f['mask'][:]
    if mask.shape[2] <= channel_index:
        raise ValueError(f"Mask has {mask.shape[2]} channels; index {channel_index} is out of range.")
    return mask[:, :, channel_index]


def mask_channel_to_jpg(h5_path, output_jpg_path, channel_index=0):
    """Save the specified mask channel as a JPG image.

    Mask values are already integers (typically 0/1/2); they are clipped to
    0-255 and stored as uint8 grayscale.
    """
    try:
        channel_data = get_mask_channel(h5_path, channel_index)
        img_array = np.clip(channel_data, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        os.makedirs(os.path.dirname(output_jpg_path), exist_ok=True)
        img.save(output_jpg_path, 'JPEG', quality=95)
        print(f"Mask channel {channel_index} saved to {output_jpg_path}")
        return True
    except Exception as e:
        print(f"Error extracting mask channel: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    
    # Option 1: Convert a single channel to JPG
    h5_file = "BraTS2020_training_data/content/data/volume_1_slice_101.h5"
    output_jpg = "output/volume_1_slice_101_channel_0.jpg"

    print("Converting single channel...")
    h5_to_jpg(h5_file, output_jpg, channel_index=0, normalize=True)
    
    # # Option 2: Convert all channels to separate JPGs
    # print("\nConverting all channels...")
    # h5_to_jpg_all_channels(h5_file, "output/volume_1_slice_101", normalize=True)

    # Example: work with the mask field
    mask_arr = get_mask_channel(h5_file, channel_index=0)
    print(f"Mask channel shape: {mask_arr.shape}, unique values: {np.unique(mask_arr)}")
    mask_output = "output/volume_1_slice_101_mask0.jpg"
    mask_channel_to_jpg(h5_file, mask_output, channel_index=0)

