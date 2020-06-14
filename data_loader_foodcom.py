from __future__ import print_function
import torch.utils.data as data
import os
import pickle
import numpy as np
import lmdb
import torch

class TextLoader(data.Dataset):
    def __init__(self, pretrained_embed_path=None, full_data=None, data_path=None, partition=None):

        if data_path == None:
            raise Exception('No data path specified.')
        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition
        with open(os.path.join(data_path, partition + '_' + full_data + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)
        with open(pretrained_embed_path, 'rb') as f:
            _ = pickle.load(f)
            self.ingre_emb = pickle.load(f)
            self.recipe_id = pickle.load(f)
            self.recipe_class = pickle.load(f)

    def __getitem__(self, index):
        rec_id = self.ids[index]
        embed_idx = self.recipe_id.tolist().index(rec_id)

        ingre = torch.from_numpy(self.ingre_emb[embed_idx])
        ingre = torch.FloatTensor(ingre)
        rec_class = self.recipe_class[embed_idx]
        return ingre, [rec_class, rec_id]

    def __len__(self):
        return len(self.ids)

class FullTextLoader(data.Dataset):
    def __init__(self, data_path=None, full_data=None, save_tuned_embed=None):
        self.save_tuned_embed = save_tuned_embed
        if data_path == None:
            raise Exception('No data path specified.')

        self.env_train = lmdb.open(os.path.join(data_path, 'train_' + full_data + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_dev = lmdb.open(os.path.join(data_path, 'val_' + full_data + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_test = lmdb.open(os.path.join(data_path, 'test_' + full_data + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, 'train_' + full_data + '_keys.pkl'), 'rb') as f:
            ids_train = pickle.load(f)
        with open(os.path.join(data_path, 'val_' + full_data + '_keys.pkl'), 'rb') as f:
            ids_dev = pickle.load(f)
        with open(os.path.join(data_path, 'test_' + full_data + '_keys.pkl'), 'rb') as f:
            ids_test = pickle.load(f)

        self.merged_data = []

        for id in ids_train:
            index = ids_train.index(id)
            with self.env_train.begin(write=False) as txn:
                serialized_sample = txn.get(ids_train[index].encode('latin1'))
            sample = pickle.loads(serialized_sample, encoding='latin1')
            self.merged_data.append((sample, id))

        for id in ids_dev:
            index = ids_dev.index(id)
            with self.env_dev.begin(write=False) as txn:
                serialized_sample = txn.get(ids_dev[index].encode('latin1'))
            sample = pickle.loads(serialized_sample, encoding='latin1')
            self.merged_data.append((sample, id))

        for id in ids_test:
            index = ids_test.index(id)
            with self.env_test.begin(write=False) as txn:
                serialized_sample = txn.get(ids_test[index].encode('latin1'))
            sample = pickle.loads(serialized_sample, encoding='latin1')
            self.merged_data.append((sample, id))

        # use the first 20 sentence in instructions:
        self.maxInst = 20
        self.mismtch = 0.8

    def __getitem__(self, index):
        match = np.random.uniform() > self.mismtch
        target = match and 1 or -1

        sample = self.merged_data[index][0]
        rec_id = self.merged_data[index][1]
        if self.save_tuned_embed:
            target = 1
            instrs = sample['intrs']
            # print('keep instruction match with ingre ... ')
        else:
            # instructions
            if target == 1:
                instrs = sample['intrs']
            else:
                # we randomly pick one non-matching instructions
                all_idx = range(len(self.merged_data))
                rndindex = np.random.choice(all_idx)
                while rndindex == index:
                    rndindex = np.random.choice(all_idx)  # pick a random index
                instrs = self.merged_data[rndindex][0]['intrs']

        # print(instrs)
        itr_ln = len(instrs)
        if itr_ln >= self.maxInst:
            instrs = instrs[:self.maxInst][:]
            itr_ln = self.maxInst

        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        rec_class = sample['classes'] - 1

        return [instrs, itr_ln, ingrs, igr_ln], [target, rec_class, rec_id]

    def __len__(self):
        return len(self.merged_data)

class TripletLoader(data.Dataset):
    def __init__(self, data_path=None, triplet_path=None, full_data=None):

        if data_path == None:
            raise Exception('No data path specified.')

        if triplet_path is None:
            raise Exception('Unknown triplet scoure %s.' %triplet_path)

        triplet_order = open(triplet_path, 'r').read().strip().split('\n')
        order = []
        for triplet in triplet_order:
            order.append((triplet.split('\t')[0], triplet.split('\t')[2]))

        self.env_train = lmdb.open(os.path.join(data_path, 'train_' + full_data + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_dev = lmdb.open(os.path.join(data_path, 'val_' + full_data + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_test = lmdb.open(os.path.join(data_path, 'test_' + full_data + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, 'train_' + full_data + '_keys.pkl'), 'rb') as f:
            ids_train = pickle.load(f)
        with open(os.path.join(data_path, 'val_' + full_data + '_keys.pkl'), 'rb') as f:
            ids_dev = pickle.load(f)
        with open(os.path.join(data_path, 'test_' + full_data + '_keys.pkl'), 'rb') as f:
            ids_test = pickle.load(f)

        self.sorted_data = []
        for (id, partition) in order:
            if partition == 'train':
                index = ids_train.index(id)
                with self.env_train.begin(write=False) as txn:
                    serialized_sample = txn.get(ids_train[index].encode('latin1'))

            elif partition == 'val':
                index = ids_dev.index(id)
                with self.env_dev.begin(write=False) as txn:
                    serialized_sample = txn.get(ids_dev[index].encode('latin1'))

            elif partition == 'test':
                index = ids_test.index(id)
                with self.env_test.begin(write=False) as txn:
                    serialized_sample = txn.get(ids_test[index].encode('latin1'))
            else:
                raise Exception('Unknown partition type')

            sample = pickle.loads(serialized_sample, encoding='latin1')
            self.sorted_data.append((sample, id))

        # use the first 20 sentence in instructions:
        self.maxInst = 20
        self.mismtch = 0.8

    def __getitem__(self, index):
        match = np.random.uniform() > self.mismtch
        target = match and 1 or -1

        sample = self.sorted_data[index][0]
        rec_id = self.sorted_data[index][1]

        # instructions
        if target == 1:
            instrs = sample['intrs']
        else:
            # we randomly pick one non-matching instructions
            all_idx = range(len(self.sorted_data))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index
            instrs = self.sorted_data[rndindex][0]['intrs']

        # print(instrs)
        itr_ln = len(instrs)
        if itr_ln >= self.maxInst:
            instrs = instrs[:self.maxInst][:]
            itr_ln = self.maxInst

        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        rec_class = sample['classes'] - 1

        return [instrs, itr_ln, ingrs, igr_ln], [target, rec_class, rec_id]

    def __len__(self):
        return len(self.sorted_data)