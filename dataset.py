import os
import glob
import skimage.transform as sktransform
import numpy as np
import matplotlib.image as mpimg

cameras = ['left', 'center', 'right']
camera_steering_offsets = [-0.2, 0.0, 0.2]

def crop_image(image, top_offset=.375,bottom_offset=.125):
    """
    Crop the image to remove the sky and the car hood.
    :param image: The input image to be cropped.
    :param top_offset: The fraction of the image height to crop from the top.
    :param bottom_offset: The fraction of the image height to crop from the bottom.
    :return: The cropped image.
    """
    height = image.shape[0]
    top_crop = int(height * top_offset)
    bottom_crop = int(height * (1 - bottom_offset))
    
    image = image[top_crop:bottom_crop, :, :]

    return sktransform.resize(image, (32, 128, 1))

def generate_samples(data, root_path, augmented=True):

    while True:

        indices = np.random.permutation(len(data))
        batch_size = 128
        for batch in range (0, len(indices), batch_size):
            batch_indices = indices [batch:batch + batch_size]

            x = np.empty([0, 32, 128, 1], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)

            for i in batch_indices:

                camera = np.random.randint(len(cameras)) if augmented else 1
                image = mpimg.imread(os.path.join(root_path, data[cameras[camera]].values[i].strip()))
                angle = data.steering.values[i] + camera_steering_offsets[camera]

                print(f'Image shape: {image.shape}, Steering angle: {angle}')
                print(f'Image: \n{image}')

                if augmented:
                    h, w = image.shape[:2]  
if __name__ == '__main__':
    data = sorted(glob.glob('*/*/rgb/*.jpg'))
    print(f'Found {len(data)} images')
    samples = generate_samples(data, '.', augmented=True)
    print(next(samples))