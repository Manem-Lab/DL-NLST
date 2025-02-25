import logging
import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, confusion_matrix, f1_score, accuracy_score, roc_curve, ConfusionMatrixDisplay
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
)
import argparse
from dataset.ImageDataset3D import ImageDataset3D
from models.networks_3D import choose_model


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--model', type=str, default="ResNet18",
                    choices=['ShuffleNetv1', 'ShuffleNetv2', 'ResNet18', 'ResNet50', 'ResNet101', 'ResNeXt101',
                             'SqueezeNet', 'MobileNetv1', 'MobileNetv2', 'R2Plus1D'], required=True, help='Model name.')
parser.add_argument('--image_size', type=int, default=64,
                    choices=[64, 128], help='Image size: 64 or 128')
parser.add_argument('--cross_val', action='store_true', help='Enable cross-validation.')
parser.add_argument('--pretrained', type=str, default='Kinetics',
                    choices=['Kinetics', '3DSeg_8', 'nnUNet', 'UCF101'],
                    help='Pretrained models: Kinetics, 3DSeg_8, nnUNet.')
parser.add_argument('--log_dir', type=str, default='./runs', help='Log path.')
parser.add_argument('--os', type=str, default='w_os',
                    choices=['w_os', 'wo_os'], help='With or W/O oversampling.')
args = parser.parse_args()

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

data_dir = './dataset/NLST'
label_dir = os.path.join(data_dir, 'NLST_Labels.xlsx')

batch_size = 8
num_workers = 4
max_epochs = 120
val_interval = 10

image_size = args.image_size
LR = 1e-4

cohort1_images = []
cohort2_images = []
cohort1_labels = []
cohort2_labels = []
# Load data from both sheets into separate DataFrames
df_training_set = pd.read_excel(label_dir, sheet_name='Training_Set')
df_test_set = pd.read_excel(label_dir, sheet_name='Test_Set')

# Function to construct file path based on the row information
def construct_file_path(row, cohort):
    base_path = os.path.join(data_dir, f"{cohort}/3D/")
    file_name = f"{row['PatientID']}_{row['TN']}_{row['ConvolutionKernel']}_{row['SeriesInstanceUID']}.nii.gz"
    return base_path + file_name

# Populate lists for the training set
for index, row in df_training_set.iterrows():
    cohort1_images.append(construct_file_path(row, "Cohort1"))
    cohort1_labels.append(row['Label'])
cohort1_labels = np.array(cohort1_labels)

# Populate lists for the test set
for index, row in df_test_set.iterrows():
    cohort2_images.append(construct_file_path(row, "Cohort2"))
    cohort2_labels.append(row['Label'])
cohort2_labels = np.array(cohort2_labels)
labels = np.concatenate((cohort1_labels, cohort2_labels))
num_pos = np.sum(labels == 1)
num_neg = np.sum(labels == 0)
print(f'Found {num_pos + num_neg} cases, {num_pos} positive and {num_neg} negative cases.')

transforms_list = [
    ScaleIntensity(),
    EnsureChannelFirst(),
    Resize((image_size, image_size, image_size), mode="trilinear"),
]

X_train, X_test, y_train, y_test = cohort1_images, cohort2_images, cohort1_labels, cohort2_labels
cross_val_ds = ImageDataset3D(image_files=X_train,
                              labels=torch.nn.functional.one_hot(torch.as_tensor(y_train, dtype=torch.long)),
                              transforms_list=transforms_list, is_train=True,
                              is_rgb=True if args.pretrained == 'Kinetics' else False)
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

loss_fig, loss_ax = plt.subplots()
loss_ax.set_xlabel('Epochs')
loss_ax.set_ylabel('Loss')
loss_ax.set_title(f'Training loss ({args.model})')

eval_loss_fig, eval_loss_ax = plt.subplots()
eval_loss_ax.set_xlabel('Epochs')
eval_loss_ax.set_ylabel('Loss')
eval_loss_ax.set_title(f'Evaluation loss ({args.model})')

# K-fold cross-validation
if args.cross_val:
    print('Start cross-validation!')
    for fold, (train_idx, val_idx) in enumerate(skf.split(cross_val_ds, y_train)):
        train_ds = Subset(cross_val_ds, train_idx)
        val_ds = Subset(cross_val_ds, val_idx)
        val_ds.dataset.is_train = False

        if args.os == 'w_os':
            class_counts = np.bincount(y_train[train_idx])
            class_weights = 1. / class_counts
            weights = class_weights[y_train[train_idx]]
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                      sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


        model = choose_model(args)
        model.to(device)
        loss_function = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), LR)

        epoch_loss_values = []
        best_auc = -1
        best_epoch = -1

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"K-Fold: {fold}, epoch {epoch}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0

            for batch_data in train_loader:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                step += 1

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                all_labels = []
                all_predictions = []
                all_scores = []

                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    with torch.no_grad():
                        val_outputs = model(val_images)
                        probabilities = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                        all_scores.extend(probabilities)
                        try:
                            predictions = val_outputs.argmax(dim=1)
                        except:
                            predictions = val_outputs[0].argmax(dim=1)
                        all_labels.extend(val_labels.cpu().numpy())
                        all_predictions.extend(predictions.cpu().numpy())

                all_labels = np.array(all_labels)
                all_labels = np.argmax(all_labels, axis=1)
                print(f'Converted labels: {all_labels}')
                all_predictions = np.array(all_predictions)
                print(f'Predicted labels: {all_predictions}')

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

    print(f"Cross-validation completed, at {fold}-fold, Epoch: {best_epoch}, Avg Accuracy: {np.average(acc_record):.4f}, "
          f" Avg F1 score: {np.average(f1_scores_record):.4f}, "f"Avg AUC: {np.average(auc_record):.4f}")

print('Start testing!')
train_ds = ImageDataset3D(image_files=X_train, labels=torch.nn.functional.one_hot(torch.as_tensor(y_train, dtype=torch.long)), transforms_list=transforms_list, is_train=True, is_rgb=True if args.pretrained == 'Kinetics' else False)
test_ds = ImageDataset3D(image_files=X_test, labels=torch.nn.functional.one_hot(torch.as_tensor(y_test, dtype=torch.long)), transforms_list=transforms_list, is_train=False, is_rgb=True if args.pretrained == 'Kinetics' else False)

if args.os == 'w_os':
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, sampler=sampler)
else:
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

model = choose_model(args)
model.to(device)
loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data
# loss_function = FocalLoss()

optimizer = torch.optim.Adam(model.parameters(), LR)

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
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        step += 1

    epoch_loss /= step
    train_losses.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        all_labels = []
        all_predictions = []
        all_scores = []
        step = 0
        eval_loss = 0

        for val_data in val_loader:
            step += 1
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                probabilities = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(probabilities)
                try:
                    loss = loss_function(val_outputs, val_labels.float())
                except:
                    loss = loss_function(val_outputs[0], val_labels.float())
                eval_loss += loss.item()
                try:
                    predictions = val_outputs.argmax(dim=1)
                except:
                    predictions = val_outputs[0].argmax(dim=1)
                all_labels.extend(val_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        eval_loss /= step
        eval_losses.append(eval_loss)
        all_labels = np.array(all_labels)
        all_labels = np.argmax(all_labels, axis=1)
        print(f'Converted labels: {all_labels}')
        all_predictions = np.array(all_predictions)
        print(f'Predicted labels: {all_predictions}')

        acc = accuracy_score(all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
        f1 = f1_score(all_labels, all_predictions)
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
            # torch.save(model.state_dict(), f'{writer.log_dir}/Best_Model.pth')
            print("saved new best metric model")
            np.save(f'{writer.log_dir}/best_fpr.npy', best_fpr)
            np.save(f'{writer.log_dir}/best_tpr.npy', best_tpr)

writer.close()
auc_ax.plot(best_fpr, best_tpr, label=f'Test Set - AUC: {best_auc:.4f}')
auc_ax.legend(loc="lower right")
auc_fig.tight_layout()
filename = f'{writer.log_dir}/Test_ROC_Curve.png'
auc_fig.savefig(filename)
plt.close(auc_fig)

conf_matrix = confusion_matrix(best_all_labels, best_all_predictions)
ConfusionMatrixDisplay(conf_matrix).plot(ax=con_ax)
con_ax.legend(loc="lower right")
con_fig.tight_layout()
filename = f'{writer.log_dir}/Confusion_Matrix.png'
con_fig.savefig(filename)
plt.close(con_fig)

epochs = np.arange(1, max_epochs+1)
loss_ax.plot(epochs, train_losses, label=f'Loss')
loss_ax.legend(loc="lower right")
loss_fig.tight_layout()
filename = f'{writer.log_dir}/Training_Loss.png'
loss_fig.savefig(filename)
plt.close(loss_fig)

epochs = np.arange(val_interval, max_epochs+1, val_interval)
eval_loss_ax.plot(epochs, eval_losses, label=f'Loss')
eval_loss_ax.legend(loc="lower right")
eval_loss_fig.tight_layout()
filename = f'{writer.log_dir}/Evaluation_Loss.png'
eval_loss_fig.savefig(filename)
plt.close(eval_loss_fig)

print(f"Training completed, at Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, "
      f" Best F1 score: {best_f1_score:.4f}, "f"Best AUC: {best_auc:.4f}")