from torch.utils.data import Dataset
import os
from torchvision.io import read_image


class ImSegDataSet(Dataset):
    def __init__(self,PathToDataSet,mode):
        """
        Constructor of the data set used to train the model

        parameters:
            PathToDataSet:     Location of the training data (Folder containing subfolders for images and labels)
            mode:              How is the data set used (train,validation or test)   
        """
        super().__init__()

        self.mode = mode

        #test if there is data in the given folder
        if not (os.path.exists(PathToDataSet + f"images/{self.mode}_im_0.jpg") and os.path.exists(PathToDataSet + f"labels/{self.mode}_label_0.png")):
            raise ValueError("From ImSegDataSet.__init__: Training data not availlable!")
        
        #Store the paths to the files
        self.files_X = [os.path.join(PathToDataSet + f"images/", f) for f in os.listdir(PathToDataSet + f"images/") if os.path.isfile(os.path.join(PathToDataSet + f"images/", f))]
        self.files_Y = [os.path.join(PathToDataSet + f"labels/", f) for f in os.listdir(PathToDataSet + f"labels/") if os.path.isfile(os.path.join(PathToDataSet + f"labels/", f))]

        n_X = len(self.files_X)
        n_Y = len(self.files_Y)

        #Test if there are as many features as responses
        if n_X != n_Y: raise ValueError

        #Store how many samples have been loaded 
        self.n_samples_original = n_X

        #load one image to determin the size of the data
        image = read_image(self.files_X[0])

        self.n_chanels = image.shape[0]
        self.height = image.shape[1]
        self.width = image.shape[2]

        #Print some stats about the data set
        print("#########################################################################################")
        print("INFO ABOUT THE DATA SET:")
        print(f"\tMode of the data set:\t{self.mode}")
        print(f"\tNumber of instances:\t{self.n_samples_original}")
        print(f"\tImage: (C,H,W):\t\t({self.n_chanels},{self.height},{self.width})")
        print("#########################################################################################")
        
    def __len__(self):
        """
        Returns the size of the data set.

        parameters:
            -
        
        retunrs:
            Lenght of the data set
        """
        return 10#self.n_samples_original

    def __getitem__(self, index):
        """
        Load and return one instance of the data set

        parameters:
            index:      Position of the desired instance in the data set
        
        returns:
            X:          Features of instance index
            Y:          Groundtruth labeling for instance index
        """

        #Load the image
        X = read_image(self.files_X[index]) / 255
        Y = read_image(self.files_Y[index])
    
        return X,Y
