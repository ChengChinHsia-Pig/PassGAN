from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class PasswordDataset(Dataset):
    def __init__(self, file_path, max_length):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        password = self.data[idx].strip()
        # 如果密码长度超过max_length，则截断它
        if len(password) > self.max_length:
            password = password[:self.max_length]
        # 填充密码到最大长度
        padded_password = password.ljust(self.max_length)
        return torch.tensor([ord(c) for c in padded_password], dtype=torch.float32)

# 第一次预处理数据并保存
def preprocess_and_save(file_path, max_length, save_path):
    dataset = PasswordDataset(file_path, max_length)
    
    # 使用tqdm包装DataLoader以显示进度
    loader = DataLoader(dataset, batch_size=len(dataset))
    loader = tqdm(loader, desc="Processing data")

    # 使用迭代器读取数据并保存
    for data in loader:
        torch.save(data, save_path)

# 调用函数进行预处理和保存
file_path = 'datas/train.txt'
max_length = 20
save_path = 'datas/preprocessed_data.pt'
preprocess_and_save(file_path, max_length, save_path)