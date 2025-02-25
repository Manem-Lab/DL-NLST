from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CTDataset(Dataset):
    def __init__(self, patient_images, label_dict, transform=None, is_train=True):
        self.patient_images = patient_images
        self.label_dict = label_dict
        self.patient_ids = list(patient_images.keys())
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        images = [Image.open(img_path).convert('RGB') for img_path in self.patient_images[patient_id]]
        label = self.label_dict[patient_id]

        if self.is_train:
            images = [self.transform(np.transpose(np.array(image), (2, 0, 1))) for image in images]
        else:
            images = [self.transform(np.transpose(np.array(image), (2, 0, 1))) for image in images]

        return images, label