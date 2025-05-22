"""
A series of helper functions used throughout the course.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def plot_loss_curves(train_results, val_results):

    epochs = range(len(train_results["loss_classification_train"]))

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_results['loss_classification_train'], label="loss_classification_train", color='blue')
    plt.plot(epochs, val_results['loss_classification_val'], label="loss_classification_val", color='blue', linestyle='dotted')

    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.show()


def plot_acc_curves(train_results, val_results):

    epochs = range(len(train_results["acc_classification_train"]))

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_results['acc_classification_train'], label="acc_classification_train", color='blue')
    plt.plot(epochs, val_results['acc_classification_val'], label="acc_classification_val", color='blue', linestyle='dotted')

    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.show()


def plot_t_SNE(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        perp_vals = [10],
        NUM_ITER: int = 2000
        ):
    
    model.eval()  # Set to evaluation mode
    features, labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            class_out, enc_out = model(X)  # Extract last-layer features
            features.append(enc_out.cpu().numpy())  # Move to CPU
            labels.append(y.numpy())

    features = np.concatenate(features, axis=0)  # Convert list to array
    labels = np.concatenate(labels, axis=0)

    # Apply t-SNE
    for perp in perp_vals:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=45)
        features_2d = tsne.fit_transform(features)

        # Class labels
        class_labels = {
            0: 'No DR',
            1: 'DR'
        }

        cmap = plt.cm.jet
        norm = plt.Normalize(0, 4)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, norm=norm)

        legend_elements = [
            Patch(facecolor=cmap(norm(i)), edgecolor='black', label=class_labels[i]) for i in range(2)
        ]

        # Add legend outside top-right
        plt.legend(
            handles=legend_elements,
            title="DR Stage",
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            labelspacing=1,      
            borderaxespad=0.5,    
        )

        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("t-SNE Visualization of Model's Features (Extracted by Encoder)")

        plt.tight_layout()
        plt.show()

def get_augmentation_train_transforms(num_augs, crop_size):
    return A.Compose([

        A.Resize(crop_size, crop_size),

        A.SomeOf([
            A.OpticalDistortion(distort_limit=0.3, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=40, sigma=50, p=1),
            A.Affine(scale=[1.1, 1.2], translate_percent=[-0.05, 0.05], shear=[-3, 3], rotate=[-3, 3], p=1),
            A.HorizontalFlip(p=1), 
            A.VerticalFlip(p=1), 
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),  
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  
            A.AdditiveNoise(noise_type='gaussian', spatial_mode='shared', approximation=1.0, noise_params={"mean_range": (0.0, 0.0), "std_range": (0.01, 0.02)}, p=0.3),
            A.GaussianBlur(blur_limit=1, p=0.3), 
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),  
            A.Emboss(alpha=(0.5, 0.6), strength=(0.6, 0.7), p=1),  
            A.RandomGamma(gamma_limit=(80, 120), p=1),  
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, fill_mask=None, p=1),
            ], n=num_augs, p=1),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToFloat(),
        ToTensorV2()
    ], seed=33)

def get_augmentation_no_transforms(crop_size):
    return A.Compose([
    A.Resize(crop_size, crop_size),       
    A.ToFloat(),
    ToTensorV2()], seed=33)

def get_augmentation_test_transforms(crop_size):
    return A.Compose([
        A.Resize(crop_size, crop_size),

        A.SomeOf([
            # A.OpticalDistortion(distort_limit=0.3, p=1),
            # A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            # A.ElasticTransform(alpha=40, sigma=50, p=1),
            # A.Affine(scale=[0.7, 1.4], translate_percent=[-0.05, 0.05], shear=[-15, 15], rotate=[-45, 45], p=1),
            A.HorizontalFlip(p=1), 
            A.VerticalFlip(p=1), 
            # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),  
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  
            # A.AdditiveNoise(noise_type='gaussian', spatial_mode='shared', approximation=1.0, noise_params={"mean_range": (0.0, 0.0), "std_range": (0.01, 0.02)}, p=1),
            # A.GaussianBlur(blur_limit=1, p=1), 
            # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),  
            # A.Emboss(alpha=(0.5, 0.6), strength=(0.6, 0.7), p=1),  
            # A.RandomGamma(gamma_limit=(80, 120), p=1),  
            # A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, fill_mask=None, p=1),
            ], n=0, p=1),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToFloat(),
        ToTensorV2()
    ], seed=33)




