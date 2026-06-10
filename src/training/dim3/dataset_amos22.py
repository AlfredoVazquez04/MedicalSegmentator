import os
import glob
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

class DatasetAmos22(LightningDataModule):
    
    organs = [
        'Background', 'Spleen', 'Right Kidney', 'Left Kidney', 'Gallbladder', 
        'Esophagus', 'Liver', 'Stomach', 'Aorta', 'IVC', 'Pancreas', 
        'Right Adrenal', 'Left Adrenal', 'Duodenum', 'Bladder', 'Prostate/Uterus'
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.keys = ["image", "label"]

    def setup(self, stage=None):
        self.train_data = self._get_files("imagesTr", "labelsTr")
        self.val_data = self._get_files("imagesVa", "labelsVa")
        self.test_data = self._get_files("imagesTs", "labelsTs")

        self.transform = Compose([
            LoadImaged(keys=self.keys),
            EnsureChannelFirstd(keys=self.keys),
        ])

        if stage == 'fit' or stage is None:
            self.train_ds = Dataset(data=self.train_data, transform=self.transform)
            self.val_ds = Dataset(data=self.val_data, transform=self.transform)
            
        if stage == 'test' or stage is None:
            self.test_ds = Dataset(data=self.test_data, transform=self.transform)

    def _get_files(self, img_folder, lbl_folder):
        img_path = os.path.join(self.args.data_dir, img_folder)
        lbl_path = os.path.join(self.args.data_dir, lbl_folder)
        
        images = sorted(glob.glob(os.path.join(img_path, "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(lbl_path, "*.nii.gz")))
        
        return [{self.keys[0]: i, self.keys[1]: l} for i, l in zip(images, labels)] 

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers,
            collate_fn=list_data_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers,
            collate_fn=list_data_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers,
            collate_fn=list_data_collate
        )