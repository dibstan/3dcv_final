import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import tqdm
import skimage

class ImSegDataSet(Dataset):
    def __init__(self,PathToDataSet,mode):
        """
        Constructor of the data set used to train the model

        parameters:
            PathToDataSet:
            mode:
        """
        super().__init__()

        self.mode = mode

        #test if there is data in the given folder
        if not (os.path.exists(PathToDataSet + f"images/{self.mode}_im_0.jpg") and os.path.exists(PathToDataSet + f"labels/{self.mode}_label_0.png")):
            raise ValueError("From ImSegDataSet.__init__: Training data not availlable!")
        
        #Store the paths to the files
        self.files_X = [os.path.join(PathToDataSet + f"images/", f) for f in os.listdir(PathToDataSet + f"images/") if os.path.isfile(os.path.join(PathToDataSet + f"images/", f))]
        self.files_Y = [os.path.join(PathToDataSet + f"images/", f) for f in os.listdir(PathToDataSet + f"labels/") if os.path.isfile(os.path.join(PathToDataSet + f"labels/", f))]

        n_X = len(self.files_X)
        n_Y = len(self.files_Y)

        #Test if there are as many features as responses
        if n_X != n_Y: raise ValueError

        #Store how many samples have been loaded 
        self.n_samples_original = n_X

        #load one image to determin the size of the data
        image = torch.Tensor(skimage.io.imread(self.files_X[0]))

        self.n_chanels = image.shape[2]
        self.height = image.shape[0]
        self.width = image.shape[1]
        
        #Print some stats about the data set
        print("#########################################################################################")
        print("INFO ABOUT THE DATA SET:")
        print(f"\tMode of the data set:\t{self.mode}")
        print(f"\tNumber of instances:\t{self.n_samples_original}")
        print(f"\tImage: (C,H,W):\t\t({self.n_chanels},{self.height},{self.width})")
        print("#########################################################################################")
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

DS = ImSegDataSet(PathToDataSet = "./data/test_set/",mode = "test")

from utils.image_utils import *
import matplotlib.pyplot as plt

image = skimage.io.imread(DS.files_X[0])
a = skimage.transform.rescale(image, 1 / 4, anti_aliasing=True, channel_axis=2) #Do we need anti-aliasing?

fig, ax = plt.subplots(2)
ax[0].imshow(image)
ax[1].imshow(a)
plt.show()
print(torch.tensor(image).shape,torch.tensor(a).shape)