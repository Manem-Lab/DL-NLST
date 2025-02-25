<!-- #region -->
# A benchmark of deep learning approaches to predict lung cancer risk using national lung screening trial cohort

[[Paper & Supplementary](https://www.nature.com/articles/s41598-024-84193-7)] [[Dataset & Pretrained models](https://drive.google.com/drive/folders/13EO2hUXm-rwUhlq_qgS-imChn6Ok5vvg?usp=sharing)]

## Abstract
Deep learning (DL) methods have demonstrated remarkable effectiveness in assisting with lung cancer risk prediction tasks using computed tomography (CT) scans. However, the lack of comprehensive comparison and validation of state-of-the-art (SOTA) models in practical settings limits their clinical application. This study aims to review and analyze current SOTA deep learning models for lung cancer risk prediction (malignant-benign classification). To evaluate our model’s general performance, we selected 253 out of 467 patients from a subset of the National Lung Screening Trial (NLST) who had CT scans without contrast, which are the most commonly used, and divided them into training and test cohorts. The CT scans were preprocessed into 2D-image and 3D-volume formats according to their nodule annotations. We evaluated ten 3D and eleven 2D SOTA deep learning models, which were pretrained on large-scale general-purpose datasets (Kinetics and ImageNet) and radiological datasets (3DSeg-8, nnUnet and RadImageNet), for their lung cancer risk prediction performance. Our results showed that 3D-based deep learning models generally perform better than 2D models. On the test cohort, the best-performing 3D model achieved an AUROC of 0.86, while the best 2D model reached 0.79. The lowest AUROCs for the 3D and 2D models were 0.70 and 0.62, respectively. Furthermore, pretraining on large-scale radiological image datasets did not show the expected performance advantage over pretraining on general-purpose datasets. Both 2D and 3D deep learning models can handle lung cancer risk prediction tasks effectively, although 3D models generally have superior performance than their 2D competitors. Our findings highlight the importance of carefully selecting pretrained datasets and model architectures for lung cancer risk prediction. Overall, these results have important implications for the development and clinical integration of DL-based tools in lung cancer screening.

## Installation
```
git clone https://github.com/Manem-Lab/DL-NLST
cd DL-NLST
conda create -n dl-nlst python=3.9
conda activate dl-nlst
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install monai==1.3.0
pip install ./WAMA_Modules/
pip install matplotlib pandas scikit-learn tensorboard openpyxl nibabel scikit-image
pip install numpy==1.26.4
```

## Preparation
Download the prepared dataset and pretrained models from [Google drive](https://drive.google.com/drive/folders/13EO2hUXm-rwUhlq_qgS-imChn6Ok5vvg?usp=sharing). Unzip the NLST.zip and the pretrained_weights.zip into ./dataset folder and the root path, the directory structure should look similar as follows:
```
DL-NLST/
│── dataset/
│   ├── NLST/            # Prepared dataset
│   │   ├── 2D/
│   │   ├── 3D/
│   ...
│── pretrained_weights/  # Pretrained models
│── train_2D.py
│── train_3D.py
│── README.md
...
```
Note that due to the storage space limitation, a part of 2D pretrained models can not be shared through the Google Drive. You can download them directly from Hugging Face. They are: 
```
vit_large_patch14_clip_224.openai_ft_in12k_in1k
beitv2_large_patch16_224.in1k_ft_in22k_in1k
beit_large_patch16_224.in22k_ft_in22k_in1k
caformer_b36.sail_in22k_ft_in1k
deit3_large_patch16_224.fb_in22k_ft_in1k
convformer_b36.sail_in22k_ft_in1k
swin_large_patch4_window7_224.ms_in22k_ft_in1k
volo_d4_224.sail_in1k
```
You can download their checkpoints from Hugging Face and use our training configurations to reproduce the results.

## Training for 3D models
```
python train_3D.py --model ResNet18 --cross_val
```

## Training for 2D models
```
python train_2D.py --model ResNet50 --cross_val
```
After the training and evaluation process done, the evaluation results will be available at the runs folder.

## Citation
If our work contributes to your research, please cite as follows:
```
@article{jiang2025benchmark,
  title={A benchmark of deep learning approaches to predict lung cancer risk using national lung screening trial cohort},
  author={Jiang, Yifan and Ebrahimpour, Leyla and Despr{\'e}s, Philippe and Manem, Venkata SK},
  journal={Scientific reports},
  volume={15},
  number={1},
  pages={1736},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgements
This project is developed based on the following code repositories:
1. [WAMA_Modules](https://github.com/WAMAWAMA/WAMA_Modules)
2. [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
3. [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs)

We are very grateful for their contributions to the community.

