import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.control_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123
    ):
        super(CustomDataset, self).__init__()
        self.name = name
        self.rawdata, self.scaler = self.read_data(data_root, self.name)

        self.window = window
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        self.samples = self.__getsamples(self.data, proportion, seed)
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num, self.window, self.var_num))
        for i in range(self.sample_num):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data = self.divide(x, proportion, seed)

        if self.save2npy:
            # os.makedirs('./OUTPUT/samples', exist_ok=True)
            # np.save(f"./OUTPUT/samples/{self.name}_ground_truth.npy", self.unnormalize(x))

            os.makedirs(f'./Data/samples_{ self.window}', exist_ok=True)
            np.save(f"./Data/samples_{ self.window}/{self.name}_ground_truth.npy", self.unnormalize(x))

            if self.auto_norm:
                np.save(f"./Data/samples_{ self.window}/{self.name}_norm_truth.npy", unnormalize_to_zero_to_one(x))
            else:
                np.save(f"./Data/samples_{ self.window}/{self.name}_norm_truth.npy", x)

        return train_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]

        regular_data = data[regular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    


    

class CustomDataset_irregular(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        regular_ratio=0.5, 
        save2npy=True, 
        neg_one_to_one=True,
        mode='separate', 
        distribution='geometric', 
        exclude_feats=None, 
        mean_mask_length=3, 
        masking_ratio=0.15,
        seed=123
    ):
        super(CustomDataset_irregular, self).__init__()
        self.name = name
        self.rawdata, self.scaler = self.read_data(data_root, self.name)

        self.window = window
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one
        self.mode, self.distribution, self.exclude_feats = mode, distribution, exclude_feats
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length

        self.data = self.__normalize(self.rawdata)
        self.samples = self.__getsamples(self.data)
        self.regular_samples, self.irregular_samples = self.divide(self.samples, regular_ratio, seed)
        self.train_samples = self.regular_samples
        self.irregular_samples, self.masking = self.mask_data(self.irregular_samples, seed)
        self.sample_num = self.irregular_samples.shape[0]

    def update_dataset(self, restored_data):
        self.train_samples = np.row_stack([self.regular_samples, restored_data])
    
    def __getsamples(self, data):
        x = np.zeros((self.sample_num, self.window, self.var_num))
        for i in range(self.sample_num):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        if self.save2npy:
            os.makedirs('./OUTPUT/samples', exist_ok=True)
            np.save(f"./OUTPUT/samples/{self.name}_ground_truth.npy", self.unnormalize(x))
            if self.auto_norm:
                np.save(f"./OUTPUT/samples/{self.name}_norm_truth.npy", unnormalize_to_zero_to_one(x))
            else:
                np.save(f"./OUTPUT/samples/{self.name}_norm_truth.npy", x)

        return x

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    def mask_data(self, data, seed=2023):
        masks = np.ones_like(data)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(data.shape[0]):
            x = data[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
                              self.exclude_feats)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(f"./OUTPUT/samples/{self.name}_masking.npy", masks)

            if self.auto_norm:
                np.save(f"./OUTPUT/samples/{self.name}_irregular_truth.npy", unnormalize_to_zero_to_one(data))
            else:
                np.save(f"./OUTPUT/samples/{self.name}_irregular_truth.npy", data)

        # Restore RNG.
        np.random.set_state(st0)
        return data, masks.astype(bool)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = np.setdiff1d(np.arange(size), regular_train_id, assume_unique=True)

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def __getitem__(self, ind):
        x = self.irregular_samples[ind, :, :]
        mask = self.masking[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(mask)
        
    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(name='fmri', proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

class fMRIDataset_irregular(CustomDataset_irregular):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(name='fmri', **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler