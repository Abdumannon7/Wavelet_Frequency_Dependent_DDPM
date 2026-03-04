import ddpm
import unet 
from torch.utils.data import DataLoader
from torch.utils.data   import random_split #help create batches
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import os
import torch.optim as optim
import yaml
from tqdm import tqdm
import numpy as np
import image_decomposition as dwt_transforms



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(args):

    with open(args.config_path,'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as excep:
            print(excep)  

    print(config)  

    diffusion_config=config['diffusion_params']
    dataset_config=config['dataset_params']
    model_config=config['model_params']
    train_config=config['train_params']

    #noise scheduler
    scheduler=ddpm.LinearNoiseSampler(timesteps=diffusion_config['timesteps'],
                                      beta_begin=diffusion_config['beta_begin'],
                                      beta_end=diffusion_config['beta_end'])
    


    
    #dataset call
    
    dataset=datasets.ImageFolder(root=dataset_config['path'],transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),   # even 64×64 if CPU
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
    
    train_size=int(0.8*len(dataset))
    test_size=len(dataset)-train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset,test_dataset=random_split(dataset,train_size,test_size)
    train_loader =  DataLoader(dataset=train_dataset,batch_size=train_config['batch_size'],shuffle=train_config['shuffle_bool'])


    model_LH = unet.Unet(model_config).to(device=device)
    model_LH.train()

    model_HL = unet.Unet(model_config).to(device=device)
    model_HL.train()

    model_HH = unet.Unet(model_config).to(device=device)
    model_HH.train()


    #output directories
    if not os.path.exsits(train_config['output_folder']):
        os.mkdir(train_config['output_folder'])

    #checkpoint
    if os.path.exists(os.path.join(train_config['output_name'],train_config['checkpoint_file'])) :
        print('Using checkpoint file')   
        model_LH.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device)['modelLH_state_dict'])  #check this location param again
        model_HL.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device)['modelHL_state_dict'])
        model_HH.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device)['modelHH_state_dict'])

    #train param
    num_epochs=train_config['num_epochs']
    optimizer_LH = optim.Adam(model_LH.parameters,lr=train_config['learning_rate'])
    criterion_LH =  torch.nn.MSELoss()

    optimizer_HL = optim.Adam(model_HL.parameters,lr=train_config['learning_rate'])
    criterion_HL =  torch.nn.MSELoss()

    optimizer_HH = optim.Adam(model_HH.parameters,lr=train_config['learning_rate'])
    criterion_HH =  torch.nn.MSELoss()


    #training
    for epoch_idx in range(num_epochs):
        losses_LH=[]
        losses_HL=[]
        losses_HH=[]

        for image in tqdm(train_loader):
            optimizer_LH.zero_grad()
            optimizer_HL.zero_grad()
            optimizer_HH.zero_grad()
            size=image.shape[0]
            matrix_Low, matrix_High = dwt_transforms.dwt_matrix(size)
            LL, LH, HL, HH = dwt_transforms.dwt(image,matrix_Low,matrix_High)
            # image=image.float().to(device=device)
            LH_img=LH.float().to(device=device)
            HL_img=HL.float().to(device=device)
            HH_img=HH.float().to(device=device)
            # x_hat=torch.zeros(*image.shape)  ######pass ll
            x_hat=LL
            noise=torch.randn_like(LH_img).to(device=device)

            t=torch.randint(0,diffusion_config['timesteps'],(LH_img.shape[0],)).to(device=device)
            
            noise_LH=scheduler.loss_coeff(noise,t,LH_img,x_hat)  #to calculte loss coeff
            noisy_image_LH=scheduler.added_noise(LH_img,t,noise)
            noise_pred_LH=model_LH(noisy_image_LH,t)

            loss_LH=criterion_LH(noise_pred_LH,noise_LH)
            losses_LH.append(loss_LH.item())
            loss_LH.backward()
            optimizer_LH.step()



            noise_HL=scheduler.loss_coeff(noise,t,HL_img,x_hat)  #to calculte loss coeff
            noisy_image_HL=scheduler.added_noise(HL_img,t,noise)
            noise_pred_HL=model_HL(noisy_image_HL,t)

            loss_HL=criterion_HL(noise_pred_HL,noise_HL)
            losses_HL.append(loss_HL.item())
            loss_HL.backward()
            optimizer_HL.step()



            noise_HH=scheduler.loss_coeff(noise,t,HH_img,x_hat)  #to calculte loss coeff
            noisy_image_HH=scheduler.added_noise(HH_img,t,noise)
            noise_pred_HH=model_HH(noisy_image_HH,t)

            loss_HH=criterion_HH(noise_pred_HH,noise_HH)
            losses_HH.append(loss_HH.item())
            loss_HH.backward()
            optimizer_HH.step()


        print('epoch:{} and Loss LH:{.4f} and Loss HL:{.4f} and Loss HH:{.4f}'.format(
            epoch_idx+1,np.mean(losses_LH),np.mean(losses_HL),np.mean(losses_HH)
        )) 
        torch.save({'modelLH_state_dict': model_LH.state_dict(),'modelHL_state_dict': model_HL.state_dict(),'modelHH_state_dict': model_HH.state_dict()},os.path.join(train_config['output_folder'],train_config['checkpoint_file']))   
          


     