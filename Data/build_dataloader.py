import torch
from Utils.Data_utils.sine_dataset import SineDataset, SineDataset_irreguar
from Utils.Data_utils.mujoco_dataset import MuJoCoDataset, MuJoCoDataset_irregular
from Utils.Data_utils.real_datasets import CustomDataset, fMRIDataset, CustomDataset_irregular, \
                                           fMRIDataset_irregular


def build_dataloader(
    data_root=None, 
    num_samples=None, 
    proportion=None, 
    batch_size=32, 
    name='sine', 
    irregular=False, 
    **kwargs
):
    if name == 'sine':
        if irregular:
            dataset = SineDataset_irreguar(num=num_samples, **kwargs)
        else:
            dataset = SineDataset(num=num_samples, **kwargs)
    elif name == 'mujoco':
        if irregular:
            dataset = MuJoCoDataset_irregular(num=num_samples, **kwargs)
        else:
            dataset = MuJoCoDataset(num=num_samples, **kwargs)
    elif name == 'fmri':
        if irregular:
            dataset = fMRIDataset_irregular(data_root=data_root, proportion=proportion, **kwargs)
        else:
            dataset = fMRIDataset(data_root=data_root, proportion=proportion, **kwargs)
    else:
        if irregular:
            dataset = CustomDataset_irregular(name=name, proportion=proportion, data_root=data_root, **kwargs)
        else:
            dataset = CustomDataset(name=name, proportion=proportion, data_root=data_root, **kwargs)
    

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=(not irregular),
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=(not irregular))
    return dataloader, dataset


def build_data(
    data_root=None, 
    num_samples=None, 
    proportion=None, 
    name='sine', 
    irregular=False, 
    **kwargs
):
    if name == 'sine':
        if irregular:
            dataset = SineDataset_irreguar(num=num_samples, **kwargs)
        else:
            dataset = SineDataset(num=num_samples, **kwargs)
            # _,b = SineDataset(num=num_samples, **kwargs).sine_data_generation()
            # print(b)
    elif name == 'mujoco':
        if irregular:
            dataset = MuJoCoDataset_irregular(num=num_samples, **kwargs)
        else:
            dataset = MuJoCoDataset(num=num_samples, **kwargs)
    elif name == 'fmri':
        if irregular:
            dataset = fMRIDataset_irregular(data_root=data_root, proportion=proportion, **kwargs)
        else:
            dataset = fMRIDataset(data_root=data_root, proportion=proportion, **kwargs)
    else:
        if irregular:
            dataset = CustomDataset_irregular(name=name, proportion=proportion, data_root=data_root, **kwargs)
        else:
            dataset = CustomDataset(name=name, proportion=proportion, data_root=data_root, **kwargs)
    return  dataset


if __name__ == '__main__':
    pass

