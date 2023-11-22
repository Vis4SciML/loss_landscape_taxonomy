import os
import h5py
from tqdm import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn import preprocessing
from torch.utils.data import TensorDataset


FEATURE_NAMES = [
    "j_zlogz",
    "j_c1_b0_mmdt",
    "j_c1_b1_mmdt",
    "j_c1_b2_mmdt",
    "j_c2_b1_mmdt",
    "j_c2_b2_mmdt",
    "j_d2_b1_mmdt",
    "j_d2_b2_mmdt",
    "j_d2_a1_b1_mmdt",
    "j_d2_a1_b2_mmdt",
    "j_m2_b1_mmdt",
    "j_m2_b2_mmdt",
    "j_n2_b1_mmdt",
    "j_n2_b2_mmdt",
    "j_mass_mmdt",
    "j_multiplicity",
]


class JetDataModule(pl.LightningDataModule):
    def __init__(self, data_file, data_dir=None, batch_size=512, num_workers=8) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size   
        self.num_workers = num_workers
        self.valid_split = 0.2  # 20%
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--data_file", type=str, default="../../jets/data.h5")
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=512)
        return parent_parser

    def load_data_dir(self, files, train_data=True):
        """
        Read and concat all h5 files in the data directory into a single
        dataframe
        """
        data = np.empty([1, 54])
        labels = np.empty([1, 5])
        files_parsed = 0
        progress_bar = tqdm(files)

        for file in progress_bar:
            if train_data:
                file = os.path.join(self.data_dir, "train", file)
            else:
                file = os.path.join(self.data_dir, "val", file)
            try:
                h5_file = h5py.File(file, "r")
                if files_parsed == 0:
                    feature_names = np.array(h5_file["jetFeatureNames"])
                    feature_names = np.array(
                        [ft.decode("utf-8") for ft in feature_names]
                    )
                    feature_indices = [
                        int(np.where(feature_names == feature)[0])
                        for feature in FEATURE_NAMES
                    ]
                h5_dataset = h5_file["jets"]
                # convert to ndarray and concatenate with dataset
                h5_dataset = np.array(h5_dataset, dtype=np.float32)
                # separate data from labels
                np_data = h5_dataset[:, :54]
                np_labels = h5_dataset[:, -6:-1]
                # update data and labels
                data = np.concatenate((data, np_data), axis=0, dtype=np.float32)
                labels = np.concatenate((labels, np_labels), axis=0, dtype=np.float32)
                h5_file.close()
                # update progress bar
                files_parsed += 1
                progress_bar.set_postfix({"files loaded": files_parsed})
            except:
                print(f"Could not load file: {file}")

        data = data[:, feature_indices]
        return data[1:].astype(np.float32), labels[1:].astype(np.float32)

    def get_data_files(self, data_dir):
        files = os.listdir(data_dir)
        files = [file for file in files if file.endswith(".h5")]
        if len(files) == 0:
            print("Directory does not contain any .h5 files")
            return None
        return files
    
    def scale_data(self):
        scaler = preprocessing.StandardScaler().fit(self.train_data)
        self.train_data = scaler.transform(self.train_data)
        scaler = preprocessing.StandardScaler().fit(self.val_data)
        self.val_data = scaler.transform(self.val_data)

    def save_data(self):
        # Create an HDF5 file
        print('Saving data_file', self.data_file)
        with h5py.File(self.data_file, "w") as hdf_file:
            # Create datasets for "train" and "val"
            hdf_file.create_dataset("train_data", data=self.train_data)
            hdf_file.create_dataset("train_labels", data=self.train_labels)
            hdf_file.create_dataset("val_data", data=self.val_data)
            hdf_file.create_dataset("val_labels", data=self.val_labels)

    def process_data(self, save=True):
        """
        Only need to run once to prepare the data and save it  
        """
        # load train and val datasets 
        train_data_files = self.get_data_files(os.path.join(self.data_dir, "train"))
        self.train_data, self.train_labels = self.load_data_dir(train_data_files, train_data=True)
        val_data_files = self.get_data_files(os.path.join(self.data_dir, "val"))
        self.val_data, self.val_labels = self.load_data_dir(val_data_files, train_data=False)
        # process and save data to h5 file
        self.scale_data()
        if save:
            self.save_data()

    # PyTorch Lightning specific methods
    def setup(self, stage):
        """
        Load data from provided h5 data_file
        """
        with h5py.File(self.data_file, "r") as hdf_file:
            # Load data from the "train" and "labels" datasets
            self.train_data = hdf_file["train_data"][:]
            self.train_labels = hdf_file["train_labels"][:]
            self.val_data = hdf_file["val_data"][:]
            self.val_labels = hdf_file["val_labels"][:]
        
        print(f"Loaded shaped data shape (train): {self.train_data.shape}")
        print(f"Loaded shaped data datatype (train): {self.train_data.dtype}")
        print(f"Loaded shaped data shape (val): {self.val_data.shape}")
        print(f"Loaded shaped data datatype (val): {self.val_data.dtype}")

    def train_dataloader(self):
        """
        Return the training dataloader
        """
        data_tensor = torch.tensor(self.train_data, dtype=torch.float32)
        labels_tensor = torch.tensor(self.train_labels, dtype=torch.float32) 
        dataset = TensorDataset(data_tensor, labels_tensor)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Return the validation dataloader
        """
        data_tensor = torch.tensor(self.val_data, dtype=torch.float32)
        labels_tensor = torch.tensor(self.val_labels, dtype=torch.float32) 
        dataset = TensorDataset(data_tensor, labels_tensor)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Return the test dataloader
        """
        return self.val_dataloader()

    def dataloaders(self):
        """
        Return train and test as Tensor dataloaders. Used for metrics, not training
        """
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        return train_loader, val_loader
