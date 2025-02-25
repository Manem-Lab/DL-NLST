import logging
import os
import sys
import shutil
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import auc, confusion_matrix, f1_score, accuracy_score, roc_curve, ConfusionMatrixDisplay

from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    RandFlip,
)
import argparse
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
import glob
from models.networks_2D import choose_model
from dataset.ImageDataset2D import CTDataset


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--model', type=str, default="ResNet50",
                    choices=['ResNet50', 'DenseNet121', 'InceptionV3', 'ViT_Large', 'BEiTv2_Large', 'BEiTv1_Large',
                             'CAFormer_B36', 'DeiT3_Large', 'ConvFormer_B36', 'CoAtNet', 'Swin_Large', 'VOLO_D4'],
                    help='Model name.')
parser.add_argument('--image_size', type=int, default=224,
                    choices=[56, 128, 256, 224], help='Image size.')
parser.add_argument('--cross_val', action='store_true', help='Enable cross-validation.')
parser.add_argument('--log_dir', type=str, default='./runs', help='Log path.')
parser.add_argument('--pretrained', type=str, default='ImageNet',
                    choices=['RadImageNet', 'ImageNet', 'FromScratch'],
                    help='Pretrained models: RadImageNet, ImageNet or FromScratch.')
args = parser.parse_args()

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

data_dir = './dataset/NLST'
label_dir = os.path.join(data_dir, 'NLST_Labels.xlsx')

# num_workers = 8
# max_epochs = 120
# val_interval = 10

num_workers = 0
max_epochs = 3
val_interval = 1

image_size = args.image_size
batch_size = 1
LR = 1e-4

df_training_set = pd.read_excel(label_dir, sheet_name='Training_Set')
df_test_set = pd.read_excel(label_dir, sheet_name='Test_Set')
training_set_label_dict = dict(
    zip(df_training_set['PatientID'], df_training_set['Label']))
training_set_labels = df_training_set['Label']
test_set_label_dict = dict(
    zip(df_test_set['PatientID'], df_test_set['Label']))
test_set_labels = df_test_set['Label']
problem_patients = []
patient_labels = np.concatenate((training_set_labels, test_set_labels))
num_pos = np.sum(patient_labels == 1)
num_neg = np.sum(patient_labels == 0)
print(f'Found {num_pos + num_neg} cases, {num_pos} positive and {num_neg} negative cases.')

patient_images = {}
train_ids = df_training_set['PatientID'].tolist()
test_ids = df_test_set['PatientID'].tolist()
image_dir = os.path.join(data_dir, 'Cohort1', '2D')
for index, row in df_training_set.iterrows():
    image_paths = glob.glob(os.path.join(image_dir, str(row['PatientID'])+'*'))
    if row['PatientID'] not in patient_images:
        patient_images[row['PatientID']] = []
    for image_path in image_paths:
        patient_images[row['PatientID']].append(image_path)
image_dir = os.path.join(data_dir, 'Cohort2', '2D')
for index, row in df_test_set.iterrows():
    image_paths = glob.glob(os.path.join(image_dir, str(row['PatientID'])+'*'))
    if row['PatientID'] not in patient_images:
        patient_images[row['PatientID']] = []
    for image_path in image_paths:
        patient_images[row['PatientID']].append(image_path)
patient_ids = list(patient_images.keys())
label_dict = training_set_label_dict | test_set_label_dict
train_ids_label = np.array([label_dict[pid] for pid in train_ids])
test_ids_label = np.array([label_dict[pid] for pid in test_ids])

train_transforms = Compose([
    ScaleIntensity(),
    Resize((image_size, image_size), mode="bicubic"),
    RandRotate90(prob=0.5),
    RandFlip(prob=0.5, spatial_axis=0),
    RandFlip(prob=0.5, spatial_axis=1)
])
val_transforms = Compose([
    ScaleIntensity(),
    Resize((image_size, image_size), mode="bicubic")
])

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
from datetime import datetime
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join(
    "runs", args.model + '_' + str(args.pretrained) + '_' + str(args.image_size) + '_' + current_time
)
writer = SummaryWriter(log_dir=log_dir)

acc_record = []
f1_scores_record = []
auc_record = []
fpr_record = []
tpr_record = []
auc_fig, auc_ax = plt.subplots()
auc_ax.set_xlabel('False Positive Rate')
auc_ax.set_ylabel('True Positive Rate')
auc_ax.set_title(f'ROC Curve ({args.model})')

con_fig, con_ax = plt.subplots()
con_ax.set_xlabel('Predicted')
con_ax.set_ylabel('Ground true')
con_ax.set_title(f'Confusion matrix ({args.model})')

train_loss_fig, train_loss_ax = plt.subplots()
train_loss_ax.set_xlabel('Epochs')
train_loss_ax.set_ylabel('Loss')
train_loss_ax.set_title(f'Training loss ({args.model})')

eval_loss_fig, eval_loss_ax = plt.subplots()
eval_loss_ax.set_xlabel('Epochs')
eval_loss_ax.set_ylabel('Loss')
eval_loss_ax.set_title(f'Evaluation loss ({args.model})')

if args.cross_val:
    # K-fold cross-validation
    print('Start cross-validation!')
    for fold, (train_ids_idx, val_ids_idx) in enumerate(skf.split(train_ids, train_ids_label)):
        # Create training and validation datasets
        train_ds = CTDataset({pid: patient_images[pid] for pid in [train_ids[i] for i in train_ids_idx]}, label_dict, transform=train_transforms, is_train=True)
        val_ds = CTDataset({pid: patient_images[pid] for pid in [train_ids[i] for i in val_ids_idx]}, label_dict, transform=val_transforms, is_train=False)

        class_counts = np.bincount(train_ids_label[train_ids_idx])
        class_weights = 1. / class_counts
        weights = class_weights[train_ids_label[train_ids_idx]]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        model, cfg = choose_model(args)
        model.to(device)

        loss_function = torch.nn.BCEWithLogitsLoss()
        if cfg is None:
            optimizer = torch.optim.Adam(model.parameters(), LR)
            lr_scheduler = None
        else:
            for key, value in cfg.items():
                if key not in ['dataset', 'model', 'image_size', 'deploy', 'pretrained', 'log_dir', 'cross_val']:
                    setattr(args, key, value)
            optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
            if args.sched is None:
                max_epochs = args.epochs
                lr_scheduler = None
            else:
                lr_scheduler, max_epochs = create_scheduler_v2(
                    optimizer,
                    **scheduler_kwargs(args),
                    updates_per_epoch=1,
            )

        # start a typical PyTorch training
        train_losses = []
        epoch_loss = 0
        best_auc = -1
        best_epoch = -1

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"K-Fold: {fold}, epoch {epoch}/{max_epochs}")
            model.train()
            step = 0

            for batch_data in train_loader:
                step += 1
                inputs = torch.stack(batch_data[0], dim=0)
                if inputs.shape[0] != inputs.shape[1]:
                    inputs = inputs.squeeze()
                else:
                    inputs = inputs.squeeze(0)
                labels = torch.nn.functional.one_hot(torch.as_tensor(batch_data[1]), 2).float()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                average_output = outputs.mean(dim=0, keepdim=True)
                try:
                    loss = loss_function(average_output, labels)
                except:
                    loss = loss_function(average_output[0], labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

            epoch_loss /= step
            train_losses.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, epoch_loss)

            if (epoch + 1) % val_interval == 0:
                # Initialize lists to store true labels and predictions
                model.eval()
                all_labels = []
                all_predictions = []
                all_scores = []

                for val_data in val_loader:
                    step += 1
                    inputs = torch.stack(val_data[0], dim=0)
                    if inputs.shape[0] != inputs.shape[1]:
                        inputs = inputs.squeeze()
                    else:
                        inputs = inputs.squeeze(0)
                    labels = torch.nn.functional.one_hot(torch.as_tensor(val_data[1]), 2).float()
                    val_images, val_labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        val_outputs = model(val_images)
                        average_output = val_outputs.mean(dim=0, keepdim=True)
                        probabilities = torch.softmax(average_output, dim=1)[:, 1].cpu().numpy()
                        all_scores.extend(probabilities)
                        try:
                            predictions = average_output.argmax(dim=1)
                        except:
                            predictions = average_output[0].argmax(dim=1)
                        all_labels.extend(val_labels.cpu().numpy())
                        all_predictions.extend(predictions.cpu().numpy())

                # Convert lists to tensors or numpy arrays
                all_labels = np.array(all_labels)
                all_labels = np.argmax(all_labels, axis=1)
                print(f'Converted labels: {all_labels}')
                all_predictions = np.array(all_predictions)
                print(f'Predicted labels: {all_predictions}')

                # Calculate precision, recall, and F1 score
                acc = accuracy_score(all_labels, all_predictions)
                fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
                f1 = f1_score(all_labels, all_predictions)
                auc_score = auc(fpr, tpr)

                if auc_score >= best_auc:
                    best_acc = acc
                    best_f1_score = f1
                    best_auc = auc_score
                    best_epoch = epoch + 1
                    best_fpr, best_tpr = fpr, tpr
                    if len(auc_record) <= fold:
                        acc_record.append(best_acc)
                        f1_scores_record.append(best_f1_score)
                        auc_record.append(best_auc)
                        fpr_record.append(best_fpr)
                        tpr_record.append(best_tpr)
                    else:
                        acc_record[fold] = best_acc
                        f1_scores_record[fold] = best_f1_score
                        auc_record[fold] = best_auc
                        fpr_record[fold] = best_fpr
                        tpr_record[fold] = best_tpr
                print(f"K-Fold: {fold}, Epoch: {epoch}，Current Accuracy: {acc:.4f}, Current F1 score: {f1:.4f} and Current AUC: {auc_score:.4f} ")
    # auc_ax.plot(np.average(np.asarray(fpr_record), axis=0), np.average(np.asarray(tpr_record), axis=0), label=f'{skf.n_splits}-Fold - Avg AUC: {np.average(auc_record):.4f}')

    print(f"Cross-validation completed, at {fold}-fold, Epoch: {best_epoch}, Avg Accuracy: {np.average(acc_record):.4f}, "
          f" Avg F1 score: {np.average(f1_scores_record):.4f}, "f"Avg AUC: {np.average(auc_record):.4f}")

print('Start testing!')
train_ds = CTDataset({pid: patient_images[pid] for pid in train_ids}, label_dict, transform=train_transforms, is_train=True)
test_ds = CTDataset({pid: patient_images[pid] for pid in test_ids}, label_dict, transform=val_transforms, is_train=False)

class_counts = np.bincount(train_ids_label)
class_weights = 1. / class_counts
weights = class_weights[train_ids_label]
sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

model, cfg = choose_model(args)
model.to(device)

loss_function = torch.nn.BCEWithLogitsLoss()

if cfg is None:
    optimizer = torch.optim.Adam(model.parameters(), LR)
    lr_scheduler = None
else:
    for key, value in cfg.items():
        if key not in ['dataset', 'model', 'image_size', 'deploy', 'pretrained', 'log_dir', 'cross_val']:
            setattr(args, key, value)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    if args.sched is None:
        max_epochs = args.epochs
        lr_scheduler = None
    else:
        lr_scheduler, max_epochs = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(args),
            updates_per_epoch=1,
        )

# start a typical PyTorch training
train_losses = []
eval_losses = []
best_auc = -1
best_epoch = -1

for epoch in range(max_epochs):
    print("-" * 10)
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs = torch.stack(batch_data[0], dim=0)
        if inputs.shape[0] != inputs.shape[1]:
            inputs = inputs.squeeze()
        else:
            inputs = inputs.squeeze(0)
        labels = torch.nn.functional.one_hot(torch.as_tensor(batch_data[1]), 2).float()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        average_output = outputs.mean(dim=0, keepdim=True)
        try:
            loss = loss_function(average_output, labels)
        except:
            loss = loss_function(average_output[0], labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    train_losses.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if lr_scheduler is not None:
        # step LR for next epoch
        lr_scheduler.step(epoch + 1, epoch_loss)

    if (epoch + 1) % val_interval == 0:
        # Initialize lists to store true labels and predictions
        model.eval()
        all_labels = []
        all_predictions = []
        all_scores = []
        step = 0
        eval_loss = 0

        for test_data in test_loader:
            step += 1
            inputs = torch.stack(test_data[0], dim=0)
            if inputs.shape[0] != inputs.shape[1]:
                inputs = inputs.squeeze()
            else:
                inputs = inputs.squeeze(0)
            labels = torch.nn.functional.one_hot(torch.as_tensor(test_data[1]), 2).float()
            val_images, val_labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                average_output = val_outputs.mean(dim=0, keepdim=True)
                try:
                    loss = loss_function(average_output, val_labels)
                except:
                    loss = loss_function(average_output[0], val_labels)
                eval_loss += loss.item()
                probabilities = torch.softmax(average_output, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(probabilities)
                try:
                    predictions = average_output.argmax(dim=1)
                except:
                    predictions = average_output[0].argmax(dim=1)
                all_labels.extend(val_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        eval_loss /= step
        eval_losses.append(eval_loss)

        # Convert lists to tensors or numpy arrays
        all_labels = np.array(all_labels)
        all_labels = np.argmax(all_labels, axis=1)
        print(f'Converted labels: {all_labels}')
        all_predictions = np.array(all_predictions)
        print(f'Predicted labels: {all_predictions}')

        # Calculate precision, recall, and F1 score
        acc = accuracy_score(all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
        f1 = f1_score(all_labels, all_predictions)  # Handle division by zero
        auc_score = auc(fpr, tpr)

        print(
            f"Epoch: {epoch}， Current Accuracy: {acc:.4f}, Current F1 score: {f1:.4f} and Current AUC: {auc_score:.4f} ")

        if auc_score > best_auc:
            best_acc = acc
            best_f1_score = f1
            best_auc = auc_score
            best_epoch = epoch + 1
            best_fpr = fpr
            best_tpr = tpr
            best_all_labels = all_labels
            best_all_predictions = all_predictions
            torch.save(model.state_dict(), f'{writer.log_dir}/Best_Model.pth')
            print("saved new best metric model")
            np.save(f'{writer.log_dir}/best_fpr.npy', best_fpr)
            np.save(f'{writer.log_dir}/best_tpr.npy', best_tpr)

writer.close()
conf_matrix = confusion_matrix(best_all_labels, best_all_predictions)
ConfusionMatrixDisplay(conf_matrix).plot(ax=con_ax)
con_ax.legend(loc="lower right")
con_fig.tight_layout()
filename = f'{writer.log_dir}/Confusion_Matrix.png'
con_fig.savefig(filename)
plt.close(con_fig)

auc_ax.plot(best_fpr, best_tpr, label=f'Test Set - AUC: {best_auc:.4f}')
auc_ax.legend(loc="lower right")
auc_fig.tight_layout()
filename = f'{writer.log_dir}/Test_ROC_Curve.png'
auc_fig.savefig(filename)
plt.close(auc_fig)

epochs = np.arange(1, max_epochs+1)
train_loss_ax.plot(epochs, train_losses, label=f'Loss')
train_loss_ax.legend(loc="lower right")
train_loss_fig.tight_layout()
filename = f'{writer.log_dir}/Training_Loss.png'
train_loss_fig.savefig(filename)
plt.close(train_loss_fig)

epochs = np.arange(val_interval, max_epochs+1, val_interval)
eval_loss_ax.plot(epochs, eval_losses, label=f'Loss')
eval_loss_ax.legend(loc="lower right")
eval_loss_fig.tight_layout()
filename = f'{writer.log_dir}/Evaluation_Loss.png'
eval_loss_fig.savefig(filename)
plt.close(eval_loss_fig)

print(f"Training completed, at Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, "
      f" Best F1 score: {best_f1_score:.4f}, "f"Best AUC: {best_auc:.4f}")