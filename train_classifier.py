import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from decode import h5_to_imgarray

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. Dataset Definition
# ==========================================
class BraTSClassificationDataset(Dataset):
    def __init__(self, dataframe, data_root='', channel_index=3, resize_dim=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.channel_index = channel_index
        
        # Setup resizing if specified
        self.resize_transform = None
        if resize_dim is not None:
            self.resize_transform = T.Resize((resize_dim, resize_dim), antialias=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.loc[idx, 'slice_path']
        label = self.df.loc[idx, 'label']
        h5_path = os.path.join(self.data_root, rel_path) if self.data_root else rel_path
        
        # Fetch the normalized 0-255 uint8 numpy array
        img_array = h5_to_imgarray(h5_path, channel_index=self.channel_index, normalize=True)
        if img_array is None:
            img_array = np.zeros((240, 240), dtype=np.uint8)
            
        # Convert to Tensor (1, H, W) and scale to [0, 1]
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        
        # Apply resize if configured
        if self.resize_transform:
            img_tensor = self.resize_transform(img_tensor)
            
        # Normalize for ResNet (Standard mean/std for grayscale)
        img_tensor = T.Normalize(mean=[0.5], std=[0.5])(img_tensor)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)

# ===================================
# 2. Model Initialization
# =====================================
def get_resnet18_classifier(in_channels=1, num_classes=2):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# 3. Plotting
def save_plots(history, output_folder):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_curves.png'))
    plt.close()

# ======================================
# 4. Main Training Loop
# ======================================
def train_classifier(config_path):
    # Load config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)['classifier']
        
    os.makedirs(config['output_folder'], exist_ok=True)
    checkpoint_path = os.path.join(config['output_folder'], config['checkpoint_file'])

    print("Loading pre-split dataset from CSV...")
    # Load the CSV created by create_train_val_test_split
    split_csv_path = config.get('split_csv_path', 'brats_train_val_test_split.csv')
    full_df = pd.read_csv(split_csv_path)
    
    # Assign labels: 1 for significant_tumor, 0 for no_tumor
    full_df['label'] = (full_df['tumor_class'] == 'significant_tumor').astype(int)
    
    # Filter into train and validation sets based on split column
    train_df = full_df[full_df['split'] == 'train'].reset_index(drop=True)
    val_df = full_df[full_df['split'] == 'val'].reset_index(drop=True)
    
    print(f"Dataset loaded:")
    print(f"  - Training samples: {len(train_df)} (Tumor: {len(train_df[train_df['label'] == 1])}, No Tumor: {len(train_df[train_df['label'] == 0])})")
    print(f"  - Validation samples: {len(val_df)} (Tumor: {len(val_df[val_df['label'] == 1])}, No Tumor: {len(val_df[val_df['label'] == 0])})")

    # Build Datasets for Train and Validation
    train_dataset = BraTSClassificationDataset(
        dataframe=train_df, 
        data_root=config.get('data_root', ''),
        resize_dim=config.get('resize_dim')
    )
    
    val_dataset = BraTSClassificationDataset(
        dataframe=val_df, 
        data_root=config.get('data_root', ''),
        resize_dim=config.get('resize_dim')
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Setup Model, Loss, and Optimizer
    model = get_resnet18_classifier(in_channels=config['in_channels']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # State tracking
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Training Loop
    num_epochs = config['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Train Phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_bar.set_postfix(loss=loss.item())

        # Eval Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate Epoch Metrics
        t_loss = train_loss / train_total
        t_acc = train_correct / train_total
        v_loss = val_loss / val_total
        v_acc = val_correct / val_total
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1} Summary -> Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}\n")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history
        }, checkpoint_path)
        
        # Update Plots
        save_plots(history, config['output_folder'])

    print("Training Complete! Check out the training curves in the output folder.")

if __name__ == "__main__":
    train_classifier('configuration.yml')
    pass