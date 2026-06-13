import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.control_utils import noise_mask


class MuJoCoDataset(Dataset):
    def __init__(
        self, 
        window=128, 
        num=30000, 
        dim=14, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        scalar=None,
        proportion=1.
    ):
        super(MuJoCoDataset, self).__init__()
        self.window = window
        self.var_num = dim
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.rawdata, self.scaler = self._generate_random_trajectories(n_samples=num, seed=seed)
        if scalar is not None:
            self.scaler = scalar

        self.samples = self.normalize(self.rawdata)
        self.samples = self.divide(self.samples, proportion, seed)
        self.sample_num = self.samples.shape[0]

    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to generate the dataset.') from e
        
        env = suite.load('hopper', 'stand')
        physics = env.physics

		# Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
        
        data = np.zeros((n_samples, self.window, self.var_num))
        for i in range(n_samples):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)

            for t in range(self.window):
                data[i, t, :self.var_num // 2] = physics.data.qpos
                data[i, t, self.var_num // 2:] = physics.data.qvel
                physics.step()

		# Restore RNG.
        np.random.set_state(st0)

        scaler = MinMaxScaler()
        scaler = scaler.fit(data.reshape(-1, self.var_num))
        return data, scaler

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        if self.save2npy:
            # os.makedirs('./OUTPUT/samples', exist_ok=True)
            # np.save("./OUTPUT/samples/mujoco_ground_truth.npy", sq)
            os.makedirs(f'./Data/samples_{self.window}', exist_ok=True)
            np.save(f"./Data/samples_{self.window}/mujoco_ground_truth.npy", sq)
            if self.auto_norm:
                np.save(f"./Data/samples_{self.window}/mujoco_norm_truth.npy", unnormalize_to_zero_to_one(data))
            else:
                np.save(f"./Data/samples_{self.window}s/mujoco_norm_truth.npy", data)

        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

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

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class MuJoCoDataset_irregular(Dataset):
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
        scalar=None,
        regular_ratio=0.5
    ):
        super(MuJoCoDataset_irregular, self).__init__()
        self.window = window
        self.var_num = dim
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.mode, self.distribution, self.exclude_feats = mode, distribution, exclude_feats
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length

        self.rawdata, self.scaler = self._generate_random_trajectories(n_samples=num, seed=seed)
        if scalar is not None:
            self.scaler = scalar

        self.samples = self.normalize(self.rawdata)
        self.regular_samples, self.irregular_samples = self.divide(self.samples, regular_ratio, seed)
        self.train_samples = self.regular_samples
        self.irregular_samples, self.masking = self.mask_data(self.irregular_samples, seed)
        self.sample_num = self.irregular_samples.shape[0]

    def update_dataset(self, restored_data):
        self.train_samples = np.row_stack([self.regular_samples, restored_data])
        
    def _generate_random_trajectories(self, n_samples, seed=123):

        try:
             from dm_control import suite  # noqa: F401
        except ImportError as e:
             raise Exception('Deepmind Control Suite is required to generate the dataset.') from e
        
        env = suite.load('hopper', 'stand')
        physics = env.physics

		# Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
        
        data = np.zeros((n_samples, self.window, self.var_num))
        for i in range(n_samples):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)
            for t in range(self.window):
                data[i, t, :self.var_num // 2] = physics.data.qpos
                data[i, t, self.var_num // 2:] = physics.data.qvel
                physics.step()

		# Restore RNG.
        np.random.set_state(st0)
        scaler = MinMaxScaler()
        scaler = scaler.fit(data.reshape(-1, self.var_num))

        return data, scaler

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        if self.save2npy:
            os.makedirs('./OUTPUT/samples', exist_ok=True)
            np.save("./OUTPUT/samples/mujoco_ground_truth.npy", sq)

            if self.auto_norm:
                np.save("./OUTPUT/samples/mujoco_norm_truth.npy", unnormalize_to_zero_to_one(data))
            else:
                np.save("./OUTPUT/samples/mujoco_norm_truth.npy", data)

        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

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
            np.save("./OUTPUT/samples/mujoco_masking.npy", masks)

            if self.auto_norm:
                np.save("./OUTPUT/samples/mujoco_irregular_truth.npy", unnormalize_to_zero_to_one(data))
            else:
                np.save("./OUTPUT/samples/mujoco_irregular_truth.npy", data)

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

    def __getitem__(self, ind):
        x = self.irregular_samples[ind, :, :]
        mask = self.masking[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(mask)
        
    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return self.sample_num