import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


from decode import h5_to_imgarray

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Test Dataset Definition
# ==========================================
class BraTSTestDataset(Dataset):
    def __init__(self, dataframe, data_root='', channel_index=3, resize_dim=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.channel_index = channel_index
        
        self.resize_transform = None
        if resize_dim is not None:
            self.resize_transform = T.Resize((resize_dim, resize_dim), antialias=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.loc[idx, 'slice_path']
        tumor_class_str = self.df.loc[idx, 'tumor_class']
        
        # Map string label to binary: 'no_tumor' -> 0, anything else -> 1
        label = 0 if tumor_class_str == 'no_tumor' else 1
        
        h5_path = os.path.join(self.data_root, rel_path) if self.data_root else rel_path
        
        # Load and normalize
        img_array = h5_to_imgarray(h5_path, channel_index=self.channel_index, normalize=True)
        if img_array is None:
            img_array = np.zeros((240, 240), dtype=np.uint8)
            
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        
        if self.resize_transform:
            img_tensor = self.resize_transform(img_tensor)
            
        img_tensor = T.Normalize(mean=[0.5], std=[0.5])(img_tensor)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)

# ========================================
# 2. Model Loading Helper
# ========================================
def load_resnet18(checkpoint_path, in_channels=1, num_classes=2):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Check if we saved the whole state dict or just the model state
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

# ==================================
# 3. Main Evaluation Function
# ==================================
def evaluate_model(csv_path, checkpoint_path, data_root='', resize_dim=128, batch_size=32):
    print("Loading test data...")
    df = pd.read_csv(csv_path)
    
    # Filter strictly for the test set
    test_df = df[df['split'] == 'test'].copy()
    print(f"Found {len(test_df)} samples in the test split.")
    
    if len(test_df) == 0:
        print("Error: No test samples found. Check your CSV 'split' column.")
        return
        
    # Build DataLoader
    test_dataset = BraTSTestDataset(dataframe=test_df, data_root=data_root, resize_dim=resize_dim)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load Model
    print(f"Loading model checkpoint from {checkpoint_path}...")
    model = load_resnet18(checkpoint_path, in_channels=1)
    
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.numpy()
            
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets)
            
    # Calculate Metrics
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    print("\n" + "="*50)
    print(" MODEL EVALUATION RESULTS ")
    print("="*50)
    
    # Classification Report (Precision, Recall, F1)
    target_names = ['No Tumor (0)', 'Tumor (1)']
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]:<5} | FP: {cm[0][1]:<5}")
    print(f"FN: {cm[1][0]:<5} | TP: {cm[1][1]:<5}")
    print("="*50)

if __name__ == "__main__":
    # Load configuration from YAML
    config_path = 'configuration.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)['classifier']
    
    CSV_PATH = config.get('split_csv_path', 'brats_train_val_test_split.csv')
    CHECKPOINT_PATH = os.path.join(config['output_folder'], config['checkpoint_file'])
    DATA_ROOT = config.get('data_root', '')
    RESIZE_DIM = config.get('resize_dim', 128)
    BATCH_SIZE = config.get('batch_size', 32)
    
    print(f"Configuration loaded from {config_path}")
    print(f"  CSV Path: {CSV_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Data Root: {DATA_ROOT}")
    print(f"  Resize Dim: {RESIZE_DIM}")
    print(f"  Batch Size: {BATCH_SIZE}\n")
    
    # Ensure resize_dim matches what you used during training!
    evaluate_model(CSV_PATH, CHECKPOINT_PATH, DATA_ROOT, resize_dim=RESIZE_DIM, batch_size=BATCH_SIZE)