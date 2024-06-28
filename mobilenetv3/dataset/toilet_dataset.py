import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler


class ToiletSymbolDataset(Dataset):
    def __init__(self, data_root):

        self.symbol_pairs = []


    def __len__(self):
        return len(self.symbol_pairs)

    def __getitem__(self, item):
        self.symbol_pairs[item]


# class ToiletSymbolSampler(RandomSampler):
#     def __init__(self, data_source, replacement=False):
#         super(ToiletSymbolSampler, self).__init__()
#
#         self.data_source = data_source
#         # 这个参数控制的应该为是否重复采样
#         self.replacement = replacement
#         self._num_samples = num_samples
#
#     def num_samples(self):
#         # dataset size might change at runtime
#         # 初始化时不传入num_samples的时候使用数据源的长度
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples
#
#     # 返回数据集长度
#     def __len__(self):
#         return self.num_samples
#         # 索引生成
#
#     def __iter__(self):
#         n = len(self.data_source)
#         if self.replacement:
#             # 生成的随机数是可能重复的
#             return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
#         # 生成的随机数是不重复的
#         return iter(torch.randperm(n).tolist())
