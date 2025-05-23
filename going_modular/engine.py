"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from torch.utils.tensorboard import SummaryWriter

scalar = torch.amp.GradScaler('cuda', enabled=True)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn_classification: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    total_class_loss = 0

    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type=device, enabled=True):

            class_out, enc_out= model(X)

            loss_classification = loss_fn_classification(class_out, y)
    
        scalar.scale(loss_classification).backward()
        scalar.step(optimizer)
        scalar.update()

        total_class_loss += loss_classification.item()
        y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)

        total += y.size(0)
        correct += (y_pred_class == y).sum().item()

    total_class_loss = total_class_loss / len(dataloader)
    class_acc = correct / total

    return total_class_loss, class_acc

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn_classification: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    total_class_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, enc_out = model(X)
        
            # 2. Calculate and accumulate loss
            loss_classification = loss_fn_classification(class_out, y)
                
            total_class_loss += loss_classification.item()
            y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)

            total += y.size(0)
            correct += (y_pred_class == y).sum().item()

            all_preds.extend(y_pred_class.cpu())
            all_targets.extend(y.cpu())

    total_class_loss = total_class_loss / len(dataloader)

    class_acc = correct / total

    return total_class_loss, class_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    all_preds = []
    all_targets = []

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            class_out, enc_out = model(X)

            y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)

            all_preds.extend(y_pred_class.cpu())
            all_targets.extend(y.cpu())

    calculate_metrics(all_preds, all_targets)


def train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn_classification: torch.nn.Module,
    epochs: int,
    device: torch.device) -> Dict[str, List]:

    train_results = {
        "loss_classification_train": [],
        "acc_classification_train": []
        }
    
    val_results = {
        "loss_classification_val": [],
        "acc_classification_val": []
        }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(1, epochs+1)):
        loss_classification_train, acc_classification_train = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn_classification=loss_fn_classification,
            optimizer=optimizer,
            device=device)

        loss_classification_val, acc_classification_val = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn_classification=loss_fn_classification,
            device=device)
        
        scheduler.step()
        print(f"\nEpoch {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}")
        
        # Print out what's happening
        print(
            # f"Epoch: {epoch}\n"
            f"loss_classification_train: {loss_classification_train:.4f} | "
            f"loss_classification_val: {loss_classification_val:.4f} | "
            f"acc_classification_val: {acc_classification_val:.4f}"
            )

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["acc_classification_train"].append(acc_classification_train)

        val_results["loss_classification_val"].append(loss_classification_val)
        val_results["acc_classification_val"].append(acc_classification_val)

    return train_results, val_results


def calculate_metrics(y_pred_class, y):

    y_pred_class = torch.tensor(y_pred_class).numpy()
    y = torch.tensor(y).numpy()

    accuracy = accuracy_score(y, y_pred_class)

    precision, recall, f1_score, support = precision_recall_fscore_support(y, y_pred_class, average=None)

    print(f'per class precision: {precision}')
    print(f'per class recall: {recall}')
    print(f'per class f1_score: {f1_score}')

    precision, recall, f1_score, support = precision_recall_fscore_support(y, y_pred_class, average='macro')

    print(f'average scores >>> precision: {precision} | recall: {recall} | f1_score: {f1_score}')

    print(f'accuracy: {accuracy}')

    QWK = cohen_kappa_score(y, y_pred_class, weights='quadratic')

    print('QWK: ', QWK)

    cm = confusion_matrix(y, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


