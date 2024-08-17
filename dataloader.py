import pandas as pd
from PIL import Image
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image


class Dataset():
    """This code creates a dataset.pkl for training, test and validation.
        Each is a dictionary with images and captions. 
        They can be used then with image loader
        Dataset class for handling radiology images and captions.
        We can then use non radiology images
    Args:
        path (str): path for folder of ROCO dataset
        label (str): train, test or validation
        
    """

    def __init__(self, path, label):
        self.folder_path = path 
        self.label = label 
        self.radiology_images_path = os.path.join(self.folder_path, "radiology/images")
        self.caption_csv = os.path.join(self.folder_path, "radiology/data.csv")

    def load_image_from_caption(self, file_name):
        """Load an image from the given file name and convert to numpy array.

        Args:
            file_name (strg): name of image

        Returns:
            _type_: _description_
        """
        image_path = os.path.join(self.radiology_images_path, file_name)
        try:
            with Image.open(image_path) as img:
                # Convert image to numpy array
                img_array = np.array(img)
              
                return img_array
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
    def extract(self):
        """Extract images and captions from the dataset and create the dataset dictionary
        """
       
        images = []
        captions = []
        
        try:
            df = pd.read_csv(self.caption_csv)
        except FileNotFoundError:
            print(f"CSV file not found: {self.caption_csv}")
            return
        except pd.errors.EmptyDataError:
            print("CSV file is empty.")
            return
        except Exception as e:
            print(f"Error reading CSV file {self.caption_csv}: {e}")
            return
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
            file_name = row.get('name')
            caption = row.get('caption')
            if not file_name or not caption:
                print(f"Missing file name or caption in row: {row}")
                continue
            
            img_array = self.load_image_from_caption(file_name)
            if img_array is not None:
                images.append(img_array)
                captions.append(caption)
        
        self.dataset = {'images': images, 'captions': captions}
        
        output_path = os.path.join("/nas-ctm01/datasets/public/ROCO/files/", f"{self.label}.pkl")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.dataset, f)
            print(f"Dataset saved to {output_path}")
        except Exception as e:
            print(f"Error saving dataset to {output_path}: {e}")
    

class RadiologyDataset(Dataset):
    """Preparation of dataset to be used with DataLoader
    Args:
        images (list): list of numpy array from .pkl files
        captions (list): list of strings from .pkl files
        tranform (transform): Pre-process the images
    """
    def __init__(self, images, captions, transform=None):
        self.images = images
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]
        # Convert image array to PIL image
        if isinstance(image, np.ndarray):
            # Handle different cases for grayscale and RGB images
            if image.ndim == 2:  # Grayscale image
                image = Image.fromarray(image).convert('RGB')
            elif image.ndim == 3 and image.shape[2] in [1, 3]:  # Grayscale with single channel or RGB
                image = Image.fromarray(image)
                if image.mode == 'L':  # Grayscale
                    image = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise TypeError("Image should be a numpy array")

        # Apply transformations (e.g., resizing)
        if self.transform:
            image = self.transform(image)

        return image, caption



        
if __name__ == '__main__':
    dataset_test=Dataset(path="/nas-ctm01/datasets/public/ROCO/test", label='test')
    dataset_test.extract()
    dataset_val=Dataset(path="/nas-ctm01/datasets/public/ROCO/validation", label='validation')
    dataset_val.extract()
    dataset_train = Dataset(path="/nas-ctm01/datasets/public/ROCO/train", label='train')
    dataset_train.extract()
