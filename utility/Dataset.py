import torch
from torch.utils.data import Dataset
import numpy as np

class RecommendDataset(Dataset):
    def __init__(self, datas, datas_user_dict, n_items, dataset_type):
        self.datas = datas
        self.datas_user_dict = datas_user_dict
        self.dataset_type = dataset_type
        self.n_items = n_items

    def __getitem__(self, item):
        u, pos_i = self.datas[item][0], self.datas[item][1]
        if self.dataset_type == 'test':
            return torch.tensor(u), torch.tensor(pos_i)
        elif self.dataset_type == 'train':
            return torch.tensor(u), torch.tensor(pos_i), self._sample_neg(u)

    def _sample_neg(self, user):
        while True:
            neg_i_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_i_id not in self.datas_user_dict[user]:
                return torch.tensor(neg_i_id)

    def __len__(self):
        return len(self.datas)


class KGEDataset(Dataset):
    def __init__(self, all_h_list, all_r_list, all_t_list, kg_dict, n_user, n_entities):
        self.all_h_list = all_h_list
        self.all_r_list = all_r_list
        self.all_t_list = all_t_list
        self.kg_dict = kg_dict
        self.n_node = n_user + n_entities


    def __getitem__(self, item):
        h, r, pos_t = self.all_h_list[item], self.all_r_list[item], self.all_t_list[item]
        return torch.tensor(h), torch.tensor(r), torch.tensor(pos_t), self._sample_neg(h, r)

    def _sample_neg(self, h, r):
        while True:
            neg_t = np.random.randint(low=0, high=self.n_node, size=1)[0]
            if (r, neg_t) not in self.kg_dict[h]:
                return torch.tensor(neg_t)

    def __len__(self):
        return len(self.all_t_list)