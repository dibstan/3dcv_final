import shutil
import os
import numpy as np
from PIL import Image

def spliter(source_X,source_Y,val_size,test_size):
    """
    Splits the given data set into a tes, train and validation set and stores the three set at different locations

    parameters:
        source_X:       folder containing the images
        source_Y:       Folder containing the labels
        val_size:       Number of instances in the validation set
        test_size:      Number of samples in the test set

    returns:
        None
    """

    #Get files in the full set
    files_X = [f for f in os.listdir(source_X) if os.path.isfile(os.path.join(source_X, f))]
    files_Y = [f for f in os.listdir(source_Y) if os.path.isfile(os.path.join(source_Y, f))]

    n_X = len(files_X)
    n_Y = len(files_Y)

    #Check for consistency
    if n_X != n_Y: raise ValueError

    print(f"Number of instaces: {n_X}")

    #randomly select instaces for the test set and the validation set
    indices = np.random.permutation(n_X)
    indices_test = indices[:test_size]
    indices_val = indices[test_size:test_size+val_size]

    #Create folders to store the different sets
    for s in ["test","train","validation"]:
        os.makedirs(f"./data/{s}_set/images/")
        os.makedirs(f"./data/{s}_set/labels/")

    #copy the instances to the corresponding set
    counter_test = 0
    counter_val = 0
    counter_train = 0

    for i in range(n_X):
        file_path_X = source_X + files_X[i]
        file_path_Y = source_Y + files_Y[i]

        if i in indices_test:
            print(f"instance {i}:\t test set")
            shutil.copyfile(src = file_path_X, dst = f"./data/test_set/images/test_im_{counter_test}.jpg")
            shutil.copyfile(src = file_path_Y, dst = f"./data/test_set/labels/test_label_{counter_test}.png")
            counter_test += 1

        elif i in indices_val:
            print(f"instance {i}:\t validation set")
            shutil.copyfile(src = file_path_X, dst = f"./data/validation_set/images/validation_im_{counter_val}.jpg")
            shutil.copyfile(src = file_path_Y, dst = f"./data/validation_set/labels/validation_label_{counter_val}.png")
            counter_val += 1

        else:
            print(f"instance {i}:\t training set")
            shutil.copyfile(src = file_path_X, dst = f"./data/train_set/images/train_im_{counter_train}.jpg")
            shutil.copyfile(src = file_path_Y, dst = f"./data/train_set/labels/train_label_{counter_train}.png")
            counter_train += 1

def splitter_downsize(source_X,source_Y,val_size,test_size,scaling_factor):
    """
    Splits the given data set into a tes, train and validation set and stores the three set at different locations

    parameters:
        source_X:       folder containing the images
        source_Y:       Folder containing the labels
        val_size:       Number of instances in the validation set
        test_size:      Number of samples in the test set

    returns:
        None
    """

    #Get files in the full set
    files_X = [f for f in os.listdir(source_X) if os.path.isfile(os.path.join(source_X, f))]
    files_Y = [f for f in os.listdir(source_Y) if os.path.isfile(os.path.join(source_Y, f))]

    n_X = len(files_X)
    n_Y = len(files_Y)

    #Check for consistency
    if n_X != n_Y: raise ValueError

    print(f"Number of instaces: {n_X}")

    #randomly select instaces for the test set and the validation set
    indices = np.random.permutation(n_X)
    indices_test = indices[:test_size]
    indices_val = indices[test_size:test_size+val_size]

    #Create folders to store the different sets
    for s in ["test","train","validation"]:
        os.makedirs(f"./data/{s}_set/images/")
        os.makedirs(f"./data/{s}_set/labels/")

    #copy the instances to the corresponding set
    counter_test = 0
    counter_val = 0
    counter_train = 0

    for i in range(n_X):
        file_path_X = source_X + files_X[i]
        file_path_Y = source_Y + files_Y[i]

        if i in indices_test:
            print(f"instance {i}:\t test set")
            im = Image.open(file_path_X)
            im = im.resize((int(im.size[0]/scaling_factor), int(im.size[1]/scaling_factor)))
            im.save(f"./data/test_set/images/test_im_{counter_test}.jpg")
            label = Image.open(file_path_Y)
            label = label.resize((int(label.size[0]/scaling_factor), int(label.size[1]/scaling_factor)),Image.NEAREST)
            label.save(f"./data/test_set/labels/test_label_{counter_test}.png")
            counter_test += 1

        elif i in indices_val:
            print(f"instance {i}:\t validation set")
            im = Image.open(file_path_X)
            im = im.resize((int(im.size[0]/scaling_factor), int(im.size[1]/scaling_factor)))
            im.save(f"./data/validation_set/images/validation_im_{counter_val}.jpg")
            label = Image.open(file_path_Y)
            label = label.resize((int(label.size[0]/scaling_factor), int(label.size[1]/scaling_factor)),Image.NEAREST)
            label.save(f"./data/validation_set/labels/validation_label_{counter_val}.png")
            counter_val += 1

        else:
            print(f"instance {i}:\t training set")
            im = Image.open(file_path_X)
            im = im.resize((int(im.size[0]/scaling_factor), int(im.size[1]/scaling_factor)))
            im.save(f"./data/train_set/images/train_im_{counter_train}.jpg")
            label = Image.open(file_path_Y)
            label = label.resize((int(label.size[0]/scaling_factor), int(label.size[1]/scaling_factor)),Image.NEAREST)
            label.save(f"./data/train_set/labels/train_label_{counter_train}.png")
            counter_train += 1