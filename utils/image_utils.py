import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
import time

def times_divide_by_two(a, b):
    '''
    Calculate number of times a can be divided 
    by two before it is smaller than b
    '''
    count = 0
    while a >= b:
        a /= 2
        count += 1
    return count

def list_scaled_images(image, wsize, levels):
    '''
    Return list of downscaled images of length levels.
    Each image is smaller than the previous by factor two.
    First list entry is original image.
    '''
    scaled_images = [image]
    for _ in range(levels-1):
        image = skimage.transform.rescale(image, 0.5, anti_aliasing=False, channel_axis=2) #Do we need anti-aliasing?
        scaled_images.append(image)
    return scaled_images

def list_tiles(image, s):
    '''
    Return list of square tiles (size s x s)
    Number of tiles is n x m
    '''
    height, width, channels = image.shape
    n = height // s
    m = width // s 
    tiles = []
    for i in range(n):
        for j in range(m):
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
    return rotated_images)

def prepare_image(image, wsize, mirroring=False):
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
    tiles = []
    for tile in patches:
        tiles.extend(list_rotated_images(tile))
    patches = tiles
    #print(len(patches))
    return patches