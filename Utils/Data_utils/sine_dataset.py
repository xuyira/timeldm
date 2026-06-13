import os
import torch
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.control_utils import noise_mask


class SineDataset(Dataset):
    def __init__(
        self, 
        window=128, 
        num=30000, 
        dim=12, 
        save2npy=True, 
        neg_one_to_one=True, 
        seed=123,
        proportion=1.
    ):
        super(SineDataset, self).__init__()

        self.rawdata = self.sine_data_generation(no=num, seq_len=window, dim=dim, save2npy=save2npy, seed=seed)
        self.auto_norm = neg_one_to_one
        self.samples = self.normalize(self.rawdata)
        self.samples = self.divide(self.samples, proportion, seed)
        self.var_num = num
        self.sample_num = self.samples.shape[0]
        self.window = window

    def normalize(self, rawdata):
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(rawdata)
        return data

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return data
    
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
    def sine_data_generation(no, seq_len, dim, save2npy=True, seed=123):
        """Sine data generation.

        Args:
           - no: the number of samples
           - seq_len: sequence length of the time-series
           - dim: feature dimensions
    
        Returns:
           - data: generated data
        """ 
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
    
        # Initialize the output
        data = list()
        # Generate sine data
        for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
            # Initialize each time-series
            temp = list()
            # For each feature
            for k in range(dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)            
                phase = np.random.uniform(0, 0.1)
          
                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)
        
            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1)*0.5
            # Stack the generated data
            data.append(temp)

        # Restore RNG.
        np.random.set_state(st0)
        data = np.array(data)
        print('sine_data', data.shape)
        if save2npy:
            os.makedirs(f'./Data/samples_{seq_len}', exist_ok=True)
            path = f"./Data/samples_{seq_len}/sine_ground_truth.npy"
            np.save(path, data)
        return data

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class SineDataset_irreguar(Dataset):
    def __init__(
        self, 
        window=128, 
        num=30000, 
        dim=12, 
        save2npy=True, 
        neg_one_to_one=True, 
        mode='separate', 
        distribution='geometric', 
        exclude_feats=None, 
        mean_mask_length=3, 
        masking_ratio=0.15,
        seed=123,
        regular_ratio=0.5
    ):
        super(SineDataset_irreguar, self).__init__()
        self.mode, self.distribution, self.exclude_feats = mode, distribution, exclude_feats
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.save2npy = save2npy
             
        self.rawdata = self.sine_data_generation(no=num, seq_len=window, dim=dim, save2npy=save2npy, seed=seed)

        self.auto_norm = neg_one_to_one
        self.samples = self.normalize(self.rawdata)
        self.var_num = num
        self.window = window

        self.regular_samples, self.irregular_samples = self.divide(self.samples, regular_ratio, seed)
        self.train_samples = self.regular_samples
        self.irregular_samples, self.masking = self.mask_data(self.irregular_samples, seed)
        self.sample_num = self.irregular_samples.shape[0]

    def update_dataset(self, restored_data):
        self.train_samples = np.row_stack([self.regular_samples, restored_data])

    def normalize(self, rawdata):
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(rawdata)
        return data

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return data
    
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
            np.save("./OUTPUT/samples/sine_masking.npy", masks)

            if self.auto_norm:
                np.save("./OUTPUT/samples/sine_irregular_truth.npy", unnormalize_to_zero_to_one(data))
            else:
                np.save("./OUTPUT/samples/sine_irregular_truth.npy", data)

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
    def sine_data_generation(no, seq_len, dim, save2npy=True, seed=123):
        """Sine data generation.

        Args:
           - no: the number of samples
           - seq_len: sequence length of the time-series
           - dim: feature dimensions
    
        Returns:
           - data: generated data
        """ 
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
    
        # Initialize the output
        data = list()
        # Generate sine data
        for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
            # Initialize each time-series
            temp = list()
            # For each feature
            for k in range(dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)            
                phase = np.random.uniform(0, 0.1)
          
                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)
        
            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1)*0.5
            # Stack the generated data
            data.append(temp)

        # Restore RNG.
        np.random.set_state(st0)
        data = np.array(data)
        if save2npy:
            os.makedirs('./OUTPUT/samples', exist_ok=True)
            np.save("./OUTPUT/samples/sine_ground_truth.npy", data)
        return data

    def __getitem__(self, ind):
        x = self.irregular_samples[ind, :, :]
        mask = self.masking[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(mask)
    
    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return self.sample_num