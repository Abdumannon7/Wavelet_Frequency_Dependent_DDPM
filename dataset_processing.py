import os
import tqdm
import glob
import torchvision
from PIL import Image

class MRIdataset(Dataset):
    def __init__(self,tt_split,path,extension='.h5'):
        self.tt_split=tt_split
        self.extension=extension
        self.images,self.labels = self.load_images(path)

    def load_images(self,path):

        images=[]
        labels=[]

        for dir_name in tqdm(os.listdir(path)):
            for filename in glob.glob(f'{path}/{dir_name}/*.{self.extension}'):  #find all possible file names with this path
                images.append(filename)
                labels.append(int(dir_name))

        return images,labels      

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image=Image.open(self.images[item])
        image_tensor =  torchvision.transforms.ToTensor()(image)

        image_tensor=(2*image_tensor)-1

        return image_tensor
    

    def image_process():
        return 



