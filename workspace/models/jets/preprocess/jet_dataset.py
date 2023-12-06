import os
import argparse
import h5pickle as h5py
import numpy as np
from tqdm import tqdm
import yaml

from torch.utils.data import Dataset
from sklearn import preprocessing


class JetTaggingDataset(Dataset):
    def __init__(self, path, features, preprocess=None):
        """
        Args:
            path (str): Path to dataset.
            features (list[str]): Load selected features from dataset.
            preprocess (str): Standardize or normalize data.

        Raises:
            RuntimeError: If path is not a directory.
        """
        self.path = path
        self.features = features
        self.preprocess = preprocess

        if os.path.isdir(path):
            self.data, self.labels = self.load_data()
        else:
            raise RuntimeError(f"Path is not a directory: {path}")

        self.preprocess_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def preprocess_data(self):
        if preprocess == "standardize":
            scaler = preprocessing.StandardScaler().fit(self.data)
            self.data = scaler.transform(self.data)
        elif preprocess == "normalize":
            normalizer = preprocessing.Normalizer().fit(self.data)
            self.data = normalizer.transform(self.data)

    def _load_data(self, files):
        data = np.empty([1, 54])
        labels = np.empty([1, 5])
        files_parsed = 0
        progress_bar = tqdm(files)

        for file in progress_bar:
            file = os.path.join(self.path, file)
            try:
                h5_file = h5py.File(file, "r")

                if files_parsed == 0:
                    feature_names = np.array(h5_file["jetFeatureNames"])
                    feature_names = np.array(
                        [ft.decode("utf-8") for ft in feature_names]
                    )
                    feature_indices = [
                        int(np.where(feature_names == feature)[0])
                        for feature in self.features
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

    def load_data(self):
        files = os.listdir(self.path)
        files = [file for file in files if file.endswith(".h5")]
        if len(files) == 0:
            print("Directory does not contain any .h5 files")
            return None
        return self._load_data(files)


def preprocess_data(data, preprocess):
    if preprocess == "standardize":
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
    elif preprocess == "normalize":
        normalizer = preprocessing.Normalizer().fit(data)
        data = normalizer.transform(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--noise-type', type=str)
    parser.add_argument('--save-path', type=str, default='data/JT')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train_path = config['data']['train']
    test_path  = config['data']['test']
    preprocess = config['data']['preprocess']
    features   = config['data']['features']

    train_dataset = JetTaggingDataset(train_path, features, preprocess=None)
    test_dataset = JetTaggingDataset(test_path, features, preprocess=None)

    X_train, y_train = train_dataset[:]
    X_test, y_test = test_dataset[:]
    
    train_dataset_shape = X_train.shape
    test_dataset_shape = X_test.shape

    X_train = preprocess_data(X_train, preprocess)
    X_test = preprocess_data(X_test, preprocess)

    np.save(os.path.join(args.save_path, 'X_train.npy'), X_train)
    np.save(os.path.join(args.save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(args.save_path, 'X_test.npy'), X_test)
    np.save(os.path.join(args.save_path, 'y_test.npy'), y_test)
