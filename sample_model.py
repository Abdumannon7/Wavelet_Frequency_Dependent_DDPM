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
    model=unet.Unet(model_config).to(device=device)
    model.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device))
    model.eval()

    scheduler = ddpm.LinearNoiseSampler(timesteps=diffusion_config['timesteps'],
                                        beta_begin=diffusion_config['beta_begin'],
                                        beta_end=diffusion_config['beta_end'])
    
    with torch.no_grad():
        sampling(model,scheduler,train_config,model_config,diffusion_config)




    

def sampling(model,schedular,train_config,model_config,diffusion_config):

    #noise scheduler

    x_t=torch.randn((train_config['samples'],
                     model_config['image_channels'],
                     model_config['image_size'],
                     model_config['image_size'])).to(device=device)
    


    for i in tqdm(reversed(range(diffusion_config['timesteps']))):

        noise_pred=model(x_t,torch.as_tensor(i).unsqueeze(0).to(device))

        x_t,x_0_pred= schedular.sample_prev_timestep(x_t,noise_pred,torch.as_tensor(i).unsqueeze(0).to(device))


        images=torch.clamp(x_t,-1,1)
        images=(images+1)/2 #to breing back original

        grid=make_grid(images,nrow=train_config['rows'])

        image=transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['output_folder'],'samples')):
            os.mkdir(os.path.join(train_config['output_folder'],'samples'))

        image.save(os.path.join(train_config['output_folder'],'samples','x_0_{}.png'.format(i)))
        image.close() 

     