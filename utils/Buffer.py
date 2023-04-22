import torch

class ImageBuffer():
    def __init__(self,buffer_size,buffer_pick_size,C,H,W):
        """
        constructor of the buffer used to store raw images during the training of the unet

        parameters:
            buffer_size:            Maximum number of raw images in the buffer
            buffer_pick_size:       Number of images taken from the buffer in each training iteration
            C:                      Number of Channels of the images
            H:                      Height of the images
            W:                      Width of the images
        """
        self.buffer_size = buffer_size
        self.buffer_pick_size = buffer_pick_size

        self.oldest = 0
        self.size = 0

        #Case 1: One channel (p.ex. labels)
        if C == 1:
            self.internal_storage = torch.zeros(buffer_size,H,W)

        #Case 2: More than one channel (p.ex. images)
        else:
            self.internal_storage = torch.zeros(buffer_size,C,H,W)

    def sample(self,indices):
        """
        Return a tensor containing min(buffer_size,self.buffer_pick_size) raw images.

        parameters:
            -
        returns:    
            image_tensor:       Tensor containing min(buffer_size,self.buffer_pick_size) raw images.
        """

        image_tensor = self.internal_storage[indices]

        return image_tensor
    
    def update(self,new_image):
        """
        Add a new image to the buffer and remove the oldest one

        parameters:
            new_image:     Tensor of shape (C,H,W) containing the new image which should be addded to the buffer

        returns:
            None
        """

        #Case 1: The buffer is not yet filled, simply add the image
        if self.size < self.buffer_size:
            self.internal_storage[self.size] = new_image
            self.size += 1
        
        #Case 2: The buffer is completely filled, replace the oldest image by the newest one
        else:
            self.internal_storage[self.oldest] = new_image
            self.oldest = (self.oldest + 1) % self.buffer_size



        