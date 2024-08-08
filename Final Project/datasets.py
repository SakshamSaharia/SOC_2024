import h5py  # Import the h5py library to handle HDF5 file format
from torch.utils.data import Dataset  # Import Dataset class from PyTorch for creating custom datasets
import torch  # Import PyTorch for tensor operations and neural network support

# Define a custom dataset class for training data
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()  # Initialize the parent class (Dataset)
        self.h5_file = h5_file  # Store the path to the HDF5 file containing the dataset

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:  # Open the HDF5 file in read mode
            lr = f['lr'][idx]  # Retrieve the low-resolution (lr) image at index idx
            hr = f['hr'][idx]  # Retrieve the high-resolution (hr) image at index idx
            return lr, hr  # Return the lr and hr image pair

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:  # Open the HDF5 file in read mode
            return len(f['lr'])  # Return the number of samples in the dataset (length of the 'lr' dataset)


# Define a custom dataset class for evaluation data
class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()  # Initialize the parent class (Dataset)
        self.h5_file = h5_file  # Store the path to the HDF5 file containing the dataset

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:  # Open the HDF5 file in read mode
            lr = f['lr'][str(idx)][:]  # Retrieve the low-resolution (lr) image at index idx as a string
            hr = f['hr'][str(idx)][:]  # Retrieve the high-resolution (hr) image at index idx as a string
            return lr, hr  # Return the lr and hr image pair

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:  # Open the HDF5 file in read mode
            return len(f['lr'])  # Return the number of samples in the dataset (length of the 'lr' dataset)


# class EvalDataset(Dataset):
#     def __init__(self, eval_file):
#         self.eval_file = eval_file
#         self.file = h5py.File(self.eval_file, 'r')
#         self.lr_data = self.file['lr']
#         self.hr_data = self.file['hr']

#     def __getitem__(self, idx):
#         # Access the data directly without using `.value`
#         lr = torch.tensor(self.lr_data[idx], dtype=torch.float32)
#         hr = torch.tensor(self.hr_data[idx], dtype=torch.float32)
#         return lr, hr

#     def __len__(self):
#         return len(self.lr_data)

#     def __del__(self):
#         if hasattr(self, 'file'):
# #             self.file.close()

# class EvalDataset(Dataset):
#     def __init__(self, eval_file):
#         self.eval_file = eval_file
#         self.file = h5py.File(self.eval_file, 'r')
#         self.lr_data = self.file['lr']
#         self.hr_data = self.file['hr']

#     def __getitem__(self, idx):
#         # Access the data directly with string keys
#         lr = torch.tensor(self.lr_data[idx], dtype=torch.float32)
#         hr = torch.tensor(self.hr_data[idx], dtype=torch.float32)
#         return lr, hr

#     def __len__(self):
#         return len(self.lr_data)

#     def __del__(self):
#         if hasattr(self, 'file'):
#             self.file.close()

class EvalDataset2(Dataset):
    def __init__(self, eval_file):
        super(EvalDataset, self).__init__()
        self.eval_file = eval_file
        self.file = h5py.File(self.eval_file, 'r')
        self.lr_data = self.file['lr']
        self.hr_data = self.file['hr']

    def __getitem__(self, idx):
        lr = torch.tensor(self.lr_data[idx], dtype=torch.float32)
        hr = torch.tensor(self.hr_data[idx], dtype=torch.float32)
        return lr, hr

    def __len__(self):
        return len(self.lr_data)

    def __del__(self):
        self.file.close()
