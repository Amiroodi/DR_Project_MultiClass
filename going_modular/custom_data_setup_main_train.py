import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
    
def circle_crop(img, sigmaX):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 


IDRID_image_folder = "../IDRID/Imagenes/Imagenes" 
IDRID_csv_file = "../IDRID/idrid_labels.csv"  

MESSIDOR_image_folder = "../MESSIDOR/images"
MESSIDOR_csv_file = "../MESSIDOR/messidor_data.csv"

APTOS_train_image_folder = "../APTOS/resized_train_19"
APTOS_train_csv_file = "../APTOS/labels/trainLabels19.csv"  

APTOS_test_image_folder = "../APTOS/resized_test_15"
APTOS_test_csv_file = "../APTOS/labels/testLabels15.csv"  

NUM_WORKERS = 4

class LoadLabels(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file) 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 1]  # Assuming second column is label (0-4)
        return label
    
class LoadDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_file) 
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image filename and label from the DataFrame
        img_name = self.df.iloc[idx, 0]  # Assuming first column is filename
        label = self.df.iloc[idx, 1]  # Assuming second column is label (0-4)

        # Load image
        if self.image_folder == MESSIDOR_image_folder:
            img_path = os.path.join(self.image_folder, img_name)
        else:
            img_path = os.path.join(self.image_folder, img_name) + '.jpg'

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = circle_crop(image,sigmaX = 30)
        # image = cv2.addWeighted(image,4, cv2.GaussianBlur(image , (0,0) , 30) ,-4 ,128)

        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
    
def LoadDataset_train_test_split(transform, shrink_size):
    train_dataset_1 = LoadDataset(IDRID_image_folder, IDRID_csv_file, transform=transform)
    train_dataset_2 = LoadDataset(MESSIDOR_image_folder, MESSIDOR_csv_file, transform=transform)
    train_dataset_3 = LoadDataset(APTOS_train_image_folder, APTOS_train_csv_file, transform=transform)
    combined_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    # combined_dataset = train_dataset_1

    # StratifiedShuffleSplit is slow for combined_dataset because augmentations are applied, so we load labels seperately
    labels_dataset_1 = LoadLabels(IDRID_csv_file)
    labels_dataset_2 = LoadLabels(MESSIDOR_csv_file)
    labels_dataset_3 = LoadLabels(APTOS_train_csv_file)
    combined_labels_dataset = ConcatDataset([labels_dataset_1, labels_dataset_2, labels_dataset_3])
    # combined_labels_dataset = labels_dataset_1

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        combined_dataset = Subset(combined_dataset, range(shrink_size))
        combined_labels_dataset = Subset(combined_labels_dataset, range(shrink_size))

    labels = [combined_labels_dataset[i] for i in range(len(combined_labels_dataset))]  # assuming (image, label)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(range(len(combined_dataset)), labels))

    # print('train_idx: ', train_idx)
    # print('test_idx: ', test_idx)

    train_dataset = Subset(combined_dataset, train_idx)
    test_dataset = Subset(combined_dataset, test_idx)

    # for i in range(len(train_dataset)):
    #     print('train label is: ', train_dataset[i][1])
    # for i in range(len(test_dataset)):
    #     print('test label is: ', test_dataset[i][1])

    return train_dataset, test_dataset

def create_train_val_dataloader(
    train_transform,
    val_transform,
    batch_size: int, 
    shrink_size,
    num_workers: int=NUM_WORKERS
    ):
  
    # minimum augmentations should be applied to val_dataset, not the case for train_dataset
    train_dataset, _ = LoadDataset_train_test_split(transform=train_transform, shrink_size=shrink_size)
    val_dataset, _ = LoadDataset_train_test_split(transform=val_transform, shrink_size=shrink_size)

    # for i in range(len(train_dataset)):
    #     print('train: ', train_dataset[i][1])
    #     print('val: ', val_dataset[i][1])

    train_val_dataloader = []

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # train_idx and val_idx are indexes of selected items in train_dataset
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

        fold = {
                'train_dataloader': train_dataloader,
                'val_dataloader': val_dataloader
               }
        train_val_dataloader.append(fold)

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    return train_val_dataloader, class_names

def create_test_dataloader(
    test_transform,
    batch_size: int,
    shrink_size, 
    num_workers: int=NUM_WORKERS,
    ):
  
    _, test_dataset = LoadDataset_train_test_split(transform=test_transform, shrink_size=shrink_size)

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return test_dataloader, class_names

def create_train_dataloader(
    train_transform,
    batch_size: int,
    shrink_size, 
    num_workers: int=NUM_WORKERS,
    ):
    
    train_dataset, _ = LoadDataset_train_test_split(transform=train_transform, shrink_size=shrink_size)

    # Get class names
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=True)

    return train_dataloader, class_names