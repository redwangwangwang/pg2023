from __future__ import division
from __future__ import print_function

import os.path

import torch
import torchvision
from torch.utils.data import Dataset
import h5py
import numpy as np
import os.path as osp
import pickle
import random
from PIL import Image


def produce_adjacent_matrix_2_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (100, 1)
    flag_bits[99, 0] = 102

    adja_matr = np.zeros([101, 101], int)

    adja_matr[:][:] = -1e9

    adja_matr[1][1] = 0
    adja_matr[0][0] = 0
    adja_matr[1][0] = 0
    adja_matr[0][1] = 0
    # TODO
    if (flag_bits[0] == 100):
        adja_matr[1][2] = 0

    if stroke_len > 99:
        stroke_len = 99
    for idx in range(1, stroke_len):
        #
        adja_matr[idx + 1][idx + 1] = 0
        adja_matr[idx + 1][0] = 0
        adja_matr[0][idx + 1] = 0

        if (flag_bits[idx - 1] == 100):
            adja_matr[idx + 1][idx - 1 + 1] = 0

        if idx == stroke_len - 1:
            break

        if (flag_bits[idx] == 100):
            adja_matr[idx + 1][idx + 1 + 1] = 0

    return adja_matr


def produce_adjacent_matrix_4_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (100, 1)
    flag_bits[99, 0] = 102
    adja_matr = np.zeros([101, 101], int)
    adja_matr[:][:] = -1e9

    adja_matr[1][1] = 0
    adja_matr[0][0] = 0
    adja_matr[1][0] = 0
    adja_matr[0][1] = 0

    # TODO
    if stroke_len > 99:
        stroke_len = 99
    if (flag_bits[0] == 100):
        adja_matr[0 + 1][1 + 1] = 0

        #
        if (flag_bits[1] == 100):
            adja_matr[0 + 1][2 + 1] = 0

    for idx in range(1, stroke_len):
        #
        adja_matr[idx + 1][idx + 1] = 0
        adja_matr[idx + 1][0] = 0
        adja_matr[0][idx + 1] = 0

        if (flag_bits[idx - 1] == 100):
            adja_matr[idx + 1][idx - 1 + 1] = 0
            #
            if (idx >= 2) and (flag_bits[idx - 2] == 100):
                adja_matr[idx + 1][idx - 2 + 1] = 0

        if idx == stroke_len - 1:
            break

        #
        if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 100):
            adja_matr[idx + 1][idx + 1 + 1] = 0
            #
            if (idx <= (stroke_len - 3)) and (flag_bits[idx + 1] == 100):
                adja_matr[idx + 1][idx + 2 + 1] = 0

    return adja_matr


def produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (100, 1)
    flag_bits[99, 0] = 102
    adja_matr = np.zeros([101, 101], int)
    adja_matr[:][:] = -1e9

    adja_matr[1][1] = 0
    adja_matr[0][0] = 0
    adja_matr[1][0] = 0
    adja_matr[0][1] = 0
    if stroke_len > 99:
        stroke_len = 99
    adja_matr[0 + 1][stroke_len - 1 + 1] = 0
    adja_matr[stroke_len - 1 + 1][stroke_len - 1 + 1] = 0
    adja_matr[stroke_len - 1 + 1][0 + 1] = 0

    assert flag_bits[0] == 100 or flag_bits[0] == 101

    if (flag_bits[0] == 101) and stroke_len >= 2:
        adja_matr[0 + 1][1 + 1] = 0

    for idx in range(1, stroke_len):

        assert flag_bits[idx] == 100 or flag_bits[idx] == 101

        adja_matr[idx + 1][idx + 1] = 0
        adja_matr[idx + 1][0] = 0
        adja_matr[0][idx + 1] = 0

        if (flag_bits[idx - 1] == 101):
            adja_matr[idx + 1][idx - 1 + 1] = 0

        if (idx == stroke_len - 1):
            break

        #
        if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 101):
            adja_matr[idx + 1][idx + 1 + 1] = 0

    return adja_matr


'''
def check_adjacent_matrix(adjacent_matrix, stroke_len):
    assert adjacent_matrix.shape == (100, 100)
    for idx in range(1, stroke_len):
        assert adjacent_matrix[idx][idx - 1] == adjacent_matrix[idx - 1][idx]
'''


def generate_padding_mask(stroke_length):
    padding_mask = np.ones([101, 1], int)
    padding_mask[1 + stroke_length:, :] = 0
    padding_mask[0, :] = 1
    return padding_mask


class QuickDrawDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, stroke_root_dir, img_root_dir, mode, data_transforms=None):
        self.stroke_root_dir = stroke_root_dir
        self.img_root_dir = img_root_dir
        self.mode = mode
        self.data = None
        self.data_transforms = data_transforms
        with open(osp.join(stroke_root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]
        print(len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(osp.join(self.stroke_root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        self.img_mode_dir = os.path.join(self.img_root_dir, '{}'.format(self.mode))
        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        category = self.categories[cid]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        img_url = os.path.join(self.img_mode_dir, category, '{}.png'.format(sid))
        img = Image.open(img_url, 'r')
        if self.data_transforms is not None:
            try:

                img = self.data_transforms(img)
                img = img[0:3, ...]
            except:
                print("Cannot transform sketch: {}".format(img_url))
        sid_points = np.array(self.data[sketch_path][()], dtype=np.float32)
        cid = np.array(cid, dtype=np.int64)
        sample = {'points3': sid_points, 'category': cid, 'img': img}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)


def train_data_collate(batch):
    max_length = 100

    coord_list = list()
    flag_list = list()
    points3_padded_list = list()
    category_list = list()
    img_list = list()
    for item in batch:
        points3 = item['points3'][0:100, :]
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32) * 102
        points3_padded[0:points3_length, 0:2] = points3[:, 0:2]
        points3_padded[0:points3_length, 2] = points3[:, 2] + 100
        points3_padded_list.append(points3_padded)
        coord = points3_padded[:, 0:2]
        coord_list.append(coord)
        flag = points3_padded[:, 2].astype(np.int32)
        flag_list.append(flag)
        category = item['category']
        category_list.append(category)
        img_list.append(item['img'])
    # print(points3_padded_list)
    # print(coord_list)
    # print(flag_list)
    batch_padded = {
        'coord': coord_list,
        'flag': flag_list,
        'category': category_list,
    }

    batch_collate = dict()
    for k, v in batch_padded.items():
        arr = np.array(v)
        batch_collate[k] = torch.from_numpy(arr)
    batch_collate['img'] = torch.stack(img_list, dim=0)
    return batch_collate


class QuickDrawDataset_STM(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, stroke_root_dir, img_root_dir, mode, data_transforms=None):
        self.stroke_root_dir = stroke_root_dir
        self.img_root_dir = img_root_dir
        self.mode = mode
        self.data = None
        self.data_transforms = data_transforms
        with open(osp.join(stroke_root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]
        print(len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(osp.join(self.stroke_root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        self.img_mode_dir = os.path.join(self.img_root_dir, '{}'.format(self.mode))
        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        category = self.categories[cid]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        img_url = os.path.join(self.img_mode_dir, category, '{}.png'.format(sid))
        img = Image.open(img_url, 'r')
        if self.data_transforms is not None:
            try:

                img = self.data_transforms(img)
                img = img[0:3, ...]
            except:
                print("Cannot transform sketch: {}".format(img_url))
        sid_points = np.array(self.data[sketch_path][()], dtype=np.float32)

        # check_adjacent_matrix(attention_mask, stroke_len)

        cid = np.array(cid, dtype=np.int64)
        sample = {'points3': sid_points, 'category': cid, 'img': img}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)


def train_data_collate_STM(batch):
    max_length = 100

    coord_list = list()
    flag_list = list()
    points3_padded_list = list()
    category_list = list()
    img_list = list()
    attention_mask_list = list()
    attention_mask_2_list = list()
    attention_mask_3_list = list()
    padding_mask_list = list()
    for item in batch:
        points3 = item['points3'][0:100, :]
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32) * 102
        points3_padded[0:points3_length, 0:2] = points3[:, 0:2]
        points3_padded[0:points3_length, 2] = points3[:, 2] + 100
        points3_padded_list.append(points3_padded)
        coord = points3_padded[:, 0:2]
        coord_list.append(coord)
        flag = points3_padded[:, 2].astype(np.int32)
        flag_list.append(flag)
        category = item['category']
        category_list.append(category)
        img_list.append(item['img'])
        flag_bits = np.expand_dims(flag, axis=1)
        attention_mask_2_neighbors = produce_adjacent_matrix_2_neighbors(flag_bits, len(points3))

        attention_mask_4_neighbors = produce_adjacent_matrix_4_neighbors(flag_bits, len(points3))

        attention_mask_joint_neighbors = produce_adjacent_matrix_joint_neighbors(flag_bits, len(points3))
        padding_mask = generate_padding_mask(len(points3))
        attention_mask_list.append(attention_mask_2_neighbors)
        attention_mask_2_list.append(attention_mask_4_neighbors)
        attention_mask_3_list.append(attention_mask_joint_neighbors)
        padding_mask_list.append(padding_mask)

    # print(points3_padded_list)
    # print(coord_list)
    # print(flag_list)
    batch_padded = {
        'coord': coord_list,
        'flag': flag_list,
        'category': category_list,
        'attention_mask': attention_mask_list,
        'attention_mask_2': attention_mask_2_list,
        'attention_mask_3': attention_mask_3_list,
        'padding_mask': padding_mask_list
    }

    batch_collate = dict()
    for k, v in batch_padded.items():
        arr = np.array(v)
        batch_collate[k] = torch.from_numpy(arr)
    batch_collate['img'] = torch.stack(img_list, dim=0)
    # batch_collate['attention_mask'] = torch.stack(attention_mask_list, dim=0)
    # batch_collate['attention_mask_2'] = torch.stack(attention_mask_2_list, dim=0)
    # batch_collate['attention_mask_3'] = torch.stack(attention_mask_3_list, dim=0)
    # batch_collate['padding_mask'] = torch.stack(padding_mask_list, dim=0)

    return batch_collate


class QuickDrawDataset_ViT_Swin(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, stroke_root_dir, img_root_dir, mode, data_transforms=None):
        self.stroke_root_dir = stroke_root_dir
        self.img_root_dir = img_root_dir
        self.mode = mode
        self.data = None
        self.data_transforms = data_transforms
        with open(osp.join(stroke_root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]
        print(len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(osp.join(self.stroke_root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        self.img_mode_dir = os.path.join(self.img_root_dir, '{}'.format(self.mode))
        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        category = self.categories[cid]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        img_url = os.path.join(self.img_mode_dir, category, '{}.png'.format(sid))
        img = Image.open(img_url, 'r')
        if self.data_transforms is not None:
            try:

                img = self.data_transforms(img)
                img = img[0:3, ...]
            except:
                print("Cannot transform sketch: {}".format(img_url))
        cid = np.array(cid, dtype=np.int64)
        sample = {'category': cid, 'img': img}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)


if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    train_dataset = QuickDrawDataset('F:\dataset\QuickDraw_Origin\Stroke', 'F:\dataset\QuickDraw_Origin\Img', 'test',
                                     data_transforms=torchvision.transforms.ToTensor())
    a = train_dataset.__getitem__(0)
    print(torch.equal(a['img'][0], a['img'][2]))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0,
    #                                            collate_fn=train_data_collate
    #                                            )
    # for i, data in enumerate(train_loader):
    #     print(data['img'])
    #     break
