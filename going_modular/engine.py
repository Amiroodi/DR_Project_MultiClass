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
               scheduler: torch.optim.lr_scheduler,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    total_class_loss = 0

    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):

            class_out, enc_out= model(X)

            loss_classification = loss_fn_classification(class_out, y)
    
        scalar.scale(loss_classification).backward()
        scalar.step(optimizer)
        scheduler.step()
        scalar.update()

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # Calculate and accumulate accuracy metric across all batches for classification head
        total_class_loss += loss_classification.item()
        y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
        # class_acc += (y_pred_class == y).sum().item()/len(y)
        total += y.size(0)
        correct += (y_pred_class == y).sum().item()


    # Adjust metrics to get average loss and accuracy per batch 
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

    # f1_results = calculate_F1_score_multiclass(all_preds, all_targets)['f1_class']
    # print(f'f1_class: {f1_results}')

    # Adjust metrics to get average loss and accuracy per batch 
    total_class_loss = total_class_loss / len(dataloader)

    # class_acc /= len(dataloader) 
    class_acc = correct / total

    return total_class_loss, class_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn_classification: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    total_class_loss = 0

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

            # Calculate and accumulate accuracy metric across all batches for classification head
            total_class_loss += loss_classification.item()
            y_pred_class = torch.argmax(torch.softmax(class_out, dim=1), dim=1)
            # class_acc += (y_pred_class == y).sum().item()/len(y)

            # Calculate F1 score
            # Batch size should be very big so that F1 score is calculated for all test data
            # f1_results_batch = calculate_F1_score_multiclass(y_pred_class=y_pred_class.cpu(), y=y.cpu())
            # f1_results['f1_class'] += f1_results_batch['f1_class']
            all_preds.extend(y_pred_class.cpu())
            all_targets.extend(y.cpu())

    calculate_metrics(all_preds, all_targets)
    # print(f'f1_class: {f1_results}')
    
    # Adjust metrics to get average loss and accuracy per batch 
    # class_acc /= len(dataloader) 

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
            scheduler=scheduler,
            device=device)

        loss_classification_val, acc_classification_val = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn_classification=loss_fn_classification,
            device=device)
        
        # Print out what's happening
        if epoch % 5 == 0:
            print(
                f"Epoch: {epoch}\n"
                f"loss_classification_train: {loss_classification_train:.4f} | "
                f"loss_classification_validation: {loss_classification_val:.4f} | "
                f"acc_classification_validation: {acc_classification_val:.4f}"
                )

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["acc_classification_train"].append(acc_classification_train)

        val_results["loss_classification_val"].append(loss_classification_val)
        val_results["acc_classification_val"].append(acc_classification_val)

    return train_results, val_results

def pre_train(
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
    
    writer = SummaryWriter(log_dir='runs/experiment1')  # 'runs/' is the default log directory

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(0, epochs)):

        loss_classification_train, acc_classification_train = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn_classification=loss_fn_classification,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device)

        loss_classification_val, acc_classification_val = val_step(
        model=model,
        dataloader=val_dataloader,
        loss_fn_classification=loss_fn_classification,
        device=device)

        print(loss_classification_train)

        writer.add_scalar('training loss', loss_classification_train, epoch)

        # Print out what's happening
        if epoch % 4 == 0:
            print(
                f"Epoch: {epoch}\n"
                f"loss_classification_train: {loss_classification_train:.4f} | loss_classification_val: {loss_classification_val:.4f} | acc_classification_val: {acc_classification_val}"
                )

        # Update results dictionary
        train_results["loss_classification_train"].append(loss_classification_train)
        train_results["acc_classification_train"].append(acc_classification_train)

        val_results["loss_classification_val"].append(loss_classification_val)
        val_results["acc_classification_val"].append(acc_classification_val)


    return train_results, val_results

def calculate_F1_score_multiclass(y_pred_class, y, num_classes=5):

    f1_per_class = MulticlassF1Score(num_classes=num_classes, average='none')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class


    f1_results = {'f1_class_per_class': 0, 'f1_class_macro': 0}

    # print('y_pred_class', torch.tensor(y_pred_class))
    # print('y', y)

    f1_results["f1_class_per_class"] = f1_per_class(torch.tensor(y_pred_class), torch.tensor(y))
    f1_results["f1_class_macro"] = f1_macro(torch.tensor(y_pred_class), torch.tensor(y))

    print(f1_results)


def calculate_metrics(y_pred_class, y, num_classes=5):

    # f1_per_class = MulticlassF1Score(num_classes=num_classes, average='none')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class
    # f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')  # 'macro', 'micro', or 'weighted', or 'none' for F1 score for each class

    # f1_score_per_class = f1_per_class(torch.tensor(y_pred_class), torch.tensor(y))
    # f1_score_macro = f1_macro(torch.tensor(y_pred_class), torch.tensor(y))

    # for sklearn we should use numpy arrays not torch tensors
    # for sklearn we should use numpy arrays not torch tensors
    y_pred_class = torch.tensor(y_pred_class).numpy()
    y = torch.tensor(y).numpy()

    accuracy = accuracy_score(y, y_pred_class)

    precision, recall, f1_score, support = precision_recall_fscore_support(y, y_pred_class, average=None)

    print(f'per class scores: precision: {precision} | recall: {recall} | f1_score: {f1_score}')

    precision, recall, f1_score, support = precision_recall_fscore_support(y, y_pred_class, average='macro')

    print(f'average scores: precision: {precision} | recall: {recall} | f1_score: {f1_score}')

    print(f'accuracy: {accuracy}')

    QWK = cohen_kappa_score(y, y_pred_class, weights='quadratic')

    print('QWK: ', QWK)

    cm = confusion_matrix(y, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


