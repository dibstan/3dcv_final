import numpy as np
import skimage
import skimage.io
import torch

def times_divide_by_two(a, b):
    '''
    Calculate number of times a can be divided 
    by two before it is smaller than b
    '''
    count = 0
    while a >= b:
        a /= 3
        count += 1
    return count

def times_divide_by_n(a, b, n):
    '''
    Calculate number of times a can be divided 
    by n before it is smaller than b
    '''
    count = 0
    while a >= b:
        a /= n
        count += 1
    return count

def list_scaled_images(image, wsize, levels, n):
    '''
    Return list of downscaled images of length levels.
    Each image is smaller than the previous by factor n.
    First list entry is original image.
    '''
    scaled_images = [image]
    for _ in range(levels-1):
        if len(image.shape) > 2:
            image = skimage.transform.rescale(image, 1/n, anti_aliasing=False, channel_axis=2) #Do we need anti-aliasing?
        else:
            image = skimage.transform.rescale(image, 1/n, anti_aliasing=False)
        scaled_images.append(image)
    return scaled_images

def list_tiles(image, s):
    '''
    Return list of square tiles (size s x s)
    Number of tiles is n x m
    '''
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
    n = height // s
    m = width // s 
    tiles = []
    for i in range(n):
        for j in range(m):
            if channels == 1:
                tile = image[s*i:s*(i+1), s*j:s*(j+1)]
            else:
                tile = image[s*i:s*(i+1), s*j:s*(j+1), :]
            tiles.append(tile)
    return tiles

def mirror_image(image):
    '''
    Return mirrored image (mirrored along "y-axis")
    '''
    mirrored_image = image[:,::-1]
    return mirrored_image

def list_rotated_images(image):
    '''
    Return list of four rotated versions of the image.
    First entry is original image, second is rotated by 90 deg counter-clockwise, ...
    Works best with square input image
    Using np.rot90 instead of skimage function results in speedup of at least factor 1000!
    '''
    rotated_images = [image]
    for _ in range(3):
        image = np.rot90(image)
        rotated_images.append(image)
    return rotated_images

# Image prep procedure:

# Create list of downscaled images, scaling by factor 2 until smallest image side < windowsize
# Slice each image version (original and all downscaled versions) into square patches of size windowsize
# Rotate each patch three times by 90 deg to create four rotated versions of one patch
# Optional: Mirror each patch

def prepare_image(image, wsize, rotation=True, mirroring=False):
    '''
    Return list of all possible square patches for one input image.
    Number of patches is len(scaled_images) x 4 (x 2 if mirroring=True).
    If input image is not square, some parts of the image will remain unused.
    '''
    levels = times_divide_by_two(np.min([image.shape[0],image.shape[1]]), wsize)
    scaled_images = list_scaled_images(image, wsize, levels)
    #print(len(scaled_images))
    tiles = [] 
    patches = [] #tiles and patches are just two buffers for the patches until all patches are generated
    for img in scaled_images:
        tiles.extend(list_tiles(img, wsize))
    patches = tiles
    #print(len(patches))
    if mirroring == True:
        tiles = []
        for tile in patches:
            tiles.append(mirror_image(tile))
        patches.extend(tiles)
        #print(len(patches))
    if rotation == True:
        tiles = []
        for tile in patches:
            tiles.extend(list_rotated_images(tile))
        patches = tiles
    #print(len(patches))
    return patches


def prepare_image_torch(image, wsize, rotation=True, mirroring=False, n=2, use_original=True):
    '''
    Return list of all possible square patches for one input image.
    Number of patches is len(scaled_images) x 4 (x 2 if mirroring=True).
    If input image is not square, some parts of the image will remain unused.
    This function takes torch tensors as input and outputs a torch tensor

    Params:
        image: torch.tensor, either shape (dim1, dim2, 3) or shape (dim1, dim2)
        wsize: int, side length of each square patch
        rotation, mirroring: bool, triggers rotation and mirroring of each resulting patch
        n: int, factor by which the image is downscaled to create additional patches
    Returns:
        patches: torch.tensor, either shape (n_patches,3,wsize,wsize) or (n_patches,wsize,wsize)

    '''
    image = image.numpy() #Convert torch input to numpy array
    levels = times_divide_by_n(np.min([image.shape[0],image.shape[1]]), wsize, n=n)
    scaled_images = list_scaled_images(image, wsize, levels, n=n)
    #print(len(scaled_images))
    tiles = [] 
    patches = [] #tiles and patches are just two buffers for the patches until all patches are generated
    if use_original == True:
        for img in scaled_images:
            tiles.extend(list_tiles(img, wsize))
    else:
        for img in scaled_images[1:]:
            tiles.extend(list_tiles(img, wsize))
    patches = tiles
    #print(len(patches))
    if mirroring == True:
        tiles = []
        for tile in patches:
            tiles.append(mirror_image(tile))
        patches.extend(tiles)
        #print(len(patches))
    if rotation ==True:
        tiles = []
        for tile in patches:
            tiles.extend(list_rotated_images(tile))
        patches = tiles
    #patches
    patches = np.asarray(patches) #Turn list of numpy arrays into numpy array
    patches = torch.from_numpy(patches) #Convert numpy array to torch array 
    if len(patches.shape) == 4: 
        patches = patches.permute(0,3,1,2) #Permutate dimensions to fit requirements
    return patches #torch.tensor(patches)