from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import skimage.io as io
# from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np
import os

class Image_Augmentations(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = 1

    def __call__(self, raw_clean_tensor):
        # Random cropping based on specified crop size in the configuration
        raw_clean_tensor = T.RandomCrop(self.cfg['crop_size'])(raw_clean_tensor)

        # Apply random horizontal and vertical flips
        raw_clean_tensor = T.RandomHorizontalFlip()(raw_clean_tensor)
        raw_clean_tensor = T.RandomVerticalFlip()(raw_clean_tensor)

        # if self.cfg['gaussian_noise']:
        #     raw_clean_tensor[0, :-self.n_classes] = torch.abs(torch.normal(mean=raw_clean_tensor[0, :-self.n_classes], std=(self.cfg['gaussian_noise'] * raw_clean_tensor[0, :-self.n_classes])))
        
        # Randomly rotate the image by 0, 90, 180, or 270 degrees
        k = torch.randint(low=0, high=4, size=(1,))[0]
        return torch.rot90(raw_clean_tensor, k=k, dims=[-2, -1])

# def change_SNR(raw_image, clean_image, ratio='random'):
#     signal_mask = clean_image > 0
#     noise_mask = (raw_image > 0) & (clean_image == 0)
#     signal_mean = raw_image[signal_mask].mean()
#     noise_mean =  raw_image[noise_mask].mean()

#     if torch.isnan(signal_mean) or torch.isnan(noise_mean):
#         return raw_image
    
#     if ratio == 'random':
#         ratio = np.random.uniform(0.1, 1)
    
#     new_SNR_img = raw_image
#     factor = ratio * signal_mean - noise_mean
#     if factor < 0:
#         return raw_image
#     new_SNR_img[noise_mask] += torch.normal(mean=factor, std=0.5*factor, size=new_SNR_img[noise_mask].shape)
    
#     return torch.abs(new_SNR_img)

# def swap_img_patches(protein_single_imgs_tensor):
#     # Get the dimensions of the input image
#     batch_size, channels, rows, cols = protein_single_imgs_tensor.shape
    
#     for fov_idx in range(batch_size):
#         fov_imgs = protein_single_imgs_tensor[fov_idx]
#         # Check if the image size is divisible by 2
#         if rows % 2 != 0 or cols % 2 != 0:
#             raise ValueError("Image dimensions must be divisible by 2 for splitting into 2x2 patches.")

#         # Split the image into 2x2 patches
#         patches = fov_imgs.view(1, channels, rows // 2, 2, cols // 2, 2)

#         # Divide the patches into two groups
#         num_groups = 8
#         group_size = patches.shape[2] // num_groups
#         group_indices = torch.randperm(patches.shape[2])

#         # Randomly rotate only one group of patches
#         rotated_patches = torch.zeros_like(patches)

#         for i in range(num_groups):
#             group_mask = (group_indices >= i * group_size) & (group_indices < (i + 1) * group_size)
#             rotation_angle = torch.randint(4, (1,)).item()
#             rotated_patches[:, :, group_mask] = torch.rot90(patches[:, :, group_mask], k=rotation_angle, dims=(3, 5))

#         # Reshape the rotated patches back to the original shape
#         protein_single_imgs_tensor[fov_idx] = rotated_patches.view(1, channels, rows, cols)

    # return protein_single_imgs_tensor

# def min_max(image, percentile_value):
#     normalized_tensor = image / (1.1 * percentile_value)
    
#     # Cap values greater than 1
#     normalized_tensor[normalized_tensor > 1] = 1
    
#     return normalized_tensor, percentile_value

def min_max(image):
    # Calculate the percentile
    q = 0.9999  # Assuming 'percentile' is a parameter in your configuration
 
    # Calculate the q-th percentile
    perc_value = torch.kthvalue(image.flatten(), int(image.numel() * q)).values
    
    # Apply the normalization equation
    normalized_tensor = image / (1.1 * perc_value)
    
    # Cap values greater than 1
    normalized_tensor[normalized_tensor > 1] = 1
    
    return normalized_tensor, perc_value

def anscombe(image):
    # Applying the Anscombe transform
    anscombe_image = 2 * torch.sqrt(image + 3/8)
    return anscombe_image


# def MAUI_transform(image, K=25):
#     # Task 1: Create an array of coordinate pairs of the non-zero values in the image
#     coordinates = np.column_stack(np.nonzero(image))

#     # Task 2: Create another array with the same coordinate pairs but repeated according to the image values
#     repeated_coordinates = np.repeat(coordinates, image[coordinates[:, 0], coordinates[:, 1]], axis=0)

#     # Task 3: Create a list of lists of KNN of coordinates in list1 using list2
#     nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(repeated_coordinates)
#     distances, _ = nbrs.kneighbors(coordinates)
#     distances = distances.sum(axis=1) / K

#     MAUI_image = image.astype(float)
#     MAUI_image[coordinates[:, 0], coordinates[:, 1]] = distances 
#     # Convert the indices array to a list of lists

#     return MAUI_image

# def calculate_intensity_percentile(image_paths, percentile):
#     # List to store pixel values from all images
#     all_pixel_values = []
    
#     # Load each image, flatten pixel values, and add to the list
#     for image_path in image_paths:
#         img = Image.open(image_path)
#         img = img.crop((0, 48, img.width, img.height))
#         pixel_values = np.array(img).flatten()
#         all_pixel_values.extend(pixel_values)
    
#     # Calculate the percentile of the combined pixel values
#     intensity_percentile = np.percentile(all_pixel_values, percentile)
#     print(f"The {percentile}th percentile of the intensities is: {intensity_percentile}")
    
#     return intensity_percentile


class Denoising_Dataset(Dataset):
    def __init__(self, dataset_path, cfg, device, is_validation):
        self.cfg = cfg
        self.device = device
        self.data_pairs = 0
        self.input_images = []
        self.GT_images = []

        self.is_validation = is_validation
        self.n_classes = 1
        self.in_channels = 1

        # List all subdatasets names to collect data from
        if is_validation and (len(cfg['exclude_datasets']) > 0):
            sub_datasets_names = [folder for folder in os.listdir(os.path.join(dataset_path, 'clean')) if not folder.startswith(".") and folder in cfg['exclude_datasets']]
        else:
            sub_datasets_names = [folder for folder in os.listdir(os.path.join(dataset_path, 'clean')) if not folder.startswith(".") and not folder in cfg['exclude_datasets']]

        # adding all path for all the data pairs
        for sub_dataset in sub_datasets_names:
            # paths to clean and noisy data dirs
            noisy_sub_dataset_path = os.path.join(dataset_path, 'noisy', sub_dataset)
            clean_sub_dataset_path = os.path.join(dataset_path, 'clean', sub_dataset)
            
            # collect all fov folder names
            fov_folders = [folder for folder in os.listdir(noisy_sub_dataset_path) if not folder.startswith(".") and os.path.isdir(os.path.join(noisy_sub_dataset_path, folder))]

            for fov_folder in fov_folders:
                noisy_folder = os.path.join(noisy_sub_dataset_path, fov_folder, 'TIFs')
                clean_folder = os.path.join(clean_sub_dataset_path, fov_folder, 'TIFs')

                noisy_img_path = os.path.join(noisy_folder, f'{cfg["protein"]}.tif')
                clean_img_path = os.path.join(clean_folder, f'{cfg["protein"]}.tif')

                # ensures the the img appears in the clean dir as well
                if os.path.exists(clean_img_path):
                    # adding a data pair
                    self.input_images.append(noisy_img_path)
                    self.GT_images.append(clean_img_path)
                    self.data_pairs += 1

        print(f'Dataset loading completed ({self.data_pairs} data pairs)!')


    def __len__(self):
        return self.data_pairs

    def __getitem__(self, idx):
        raw_image = io.imread(self.input_images[idx])[48:,:]
        clean_image = io.imread(self.GT_images[idx])[48:,:]

        transforms = Image_Augmentations(self.cfg)
        raw_clean_stacked = [raw_image, clean_image]
        raw_clean_tensor = torch.from_numpy(np.stack(raw_clean_stacked, axis=0).astype(np.float32)).float().unsqueeze(0)
        if self.cfg['img_normalization'] == 'min_max':
            raw_clean_tensor[0,0], _ = min_max(raw_clean_tensor[0,0])
        elif self.cfg['img_normalization'] == 'anscombe':
            raw_clean_tensor[0,0] = anscombe(raw_clean_tensor[0,0])
        # if self.cfg['change_SNR'] and not self.is_validation:
        #     raw_clean_tensor[0,0] = change_SNR(raw_clean_tensor[0,0], raw_clean_tensor[0,1])
        raw_clean_tensor = transforms(raw_clean_tensor)
        # if self.cfg['swap_img_patches'] and not self.is_validation:
        #     raw_clean_tensor = swap_img_patches(raw_clean_tensor) 

        return raw_clean_tensor[0, :self.in_channels], raw_clean_tensor[0, self.in_channels:]

