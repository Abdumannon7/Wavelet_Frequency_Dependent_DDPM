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

def inference(args):


    with open(args.config_path,'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as excep:
            print(excep)  

    print(config)  

    diffusion_config=config['diffusion_params']
  
    model_config=config['model_params']
    train_config=config['train_params']

    #noise scheduler

    #load model
    
    model_LH = unet.Unet(model_config).to(device=device)
    model_HL = unet.Unet(model_config).to(device=device)
    model_HH = unet.Unet(model_config).to(device=device)
   
    # model.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device))
    model_LH.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device)['modelLH_state_dict'])  #check this location param again
    model_HL.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device)['modelHL_state_dict'])
    model_HH.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device)['modelHH_state_dict'])

    model_LH.eval()
    model_HL.eval()
    model_HH.eval()

    scheduler = ddpm.LinearNoiseSampler(timesteps=diffusion_config['timesteps'],
                                        beta_begin=diffusion_config['beta_begin'],
                                        beta_end=diffusion_config['beta_end'])
    
    with torch.no_grad():
        LL=np.zeros((120,120)) ##sample LL maybe from dataset???
        LH=sampling(model_LH,scheduler,train_config,model_config,diffusion_config,LL)
        HL=sampling(model_HL,scheduler,train_config,model_config,diffusion_config,LL)
        HH=sampling(model_HH,scheduler,train_config,model_config,diffusion_config,LL)

        low_mat,high_mat= dwt_transforms.idwt_matrix(LL.shape[0])
        image=dwt_transforms.idwt(LL,LH,HL,HH,low_mat,high_mat)

        image.save(os.path.join(train_config['output_folder'],'samples','x_0_{}.png'.format(1)))
        image.close() 







    

def sampling(model,schedular,train_config,model_config,diffusion_config,LL):

    #noise scheduler
    x_hat=LL
    x_t=torch.randn((train_config['samples'],
                     model_config['image_channels'],
                     model_config['image_size'],
                     model_config['image_size'])).to(device=device)
    


    for i in tqdm(reversed(range(diffusion_config['timesteps']))):

        noise_pred=model(x_t,torch.as_tensor(i).unsqueeze(0).to(device))

        x_t,x_0_pred= schedular.sample_prev_timestep(x_t=x_t,noise_pred=noise_pred,time=torch.as_tensor(i).unsqueeze(0).to(device),x_hat=x_hat) ###how to get LL 


        images=torch.clamp(x_t,-1,1)
        images=(images+1)/2 #to breing back original

        grid=make_grid(images,nrow=train_config['rows'])

        image=transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['output_folder'],'samples')):
            os.mkdir(os.path.join(train_config['output_folder'],'samples'))

        # image.save(os.path.join(train_config['output_folder'],'samples','x_0_{}.png'.format(i)))
        # image.close() 
        return x_t

     