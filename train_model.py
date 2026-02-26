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
    path='data/'
    dataset=datasets.ImageFolder(root=path,transform=transforms.Compose([
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


    model = unet.Unet(model_config).to(device='gpu')
    model.train()


    #output directories
    if not os.path.exsits(train_config['output_folder']):
        os.mkdir(train_config['output_folder'])

    #checkpoint
    if os.path.exists(os.path.join(train_config['output_name'],train_config['checkpoint_file'])) :
        print('Using checkpoint file')   
        model.load_state_dict(torch.load(os.path.join(train_config['output_name'],train_config['checkpoint_file']),map_location=device))  #check this location param again
    

    #train param
    num_epochs=train_config['num_epochs']
    optimizer = optim.Adam(model.parameters,lr=train_config['learning_rate'])
    criterion =  torch.nn.MSELoss()


    #training
    for epoch_idx in range(num_epochs):
        losses=[]

        for image in tqdm(train_loader):
            optimizer.zero_grad()
            image=image.float().to(device=device)

            noise=torch.randn_like(image).to(device=device)

            t=torch.randint(0,diffusion_config['timesteps'],(image.shape[0],)).to(device=device)

            noisy_image=scheduler.added_noise(image,t,noise)
            noise_pred=model(noisy_image,t)

            loss=criterion(noise_pred,noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('epoch:{} and Loss:{.4f}'.format(
            epoch_idx+1,np.mean(losses)
        ))    

def sampling(model,schedular,train_config,model_config,diffusion_config):
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

     