# !/usr/bin/env python
import random
import pickle
import numpy as np
from proc import detect_ingrs
from tqdm import *
import torchfile
import time
import utils
import os
# from ..args import get_parser
import time
import lmdb
import shutil
import sys
import json
sys.path.append("..")
from args import get_parser

def get_st(file):
    info = torchfile.load(file)

    ids = info[b'ids']
    # print(info)

    imids = []
    for i, id in enumerate(ids):
        imids.append(''.join(chr(i) for i in id))

    st_vecs = {}
    st_vecs['encs'] = info[b'encs']
    st_vecs['rlens'] = info[b'rlens']
    st_vecs['rbps'] = info[b'rbps']
    # don't use image id
    st_vecs['ids'] = imids

    print(np.shape(st_vecs['encs']), len(st_vecs['rlens']), len(st_vecs['rbps']), len(st_vecs['ids']))
    print(st_vecs['encs'][:10], st_vecs['rlens'][:10], st_vecs['rbps'][:10])
    # exit()
    return st_vecs

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

DATASET = opts.dataset

# don't use this file once dataset is clean
# removed: 40576 item
# with open('remove1M.txt','r') as f:
#     remove_ids = {w.rstrip(): i for i, w in enumerate(f)}
#     print('remove ids', remove_ids)
    # exit()

t = time.time()
print("Loading instruction skip-thought vectors...")
st_vecs_test = get_st(os.path.join(opts.sthdir, 'encs_test_1024.t7'))
st_vecs_train = get_st(os.path.join(opts.sthdir, 'encs_train_1024.t7'))
st_vecs_val = get_st(os.path.join(opts.sthdir, 'encs_val_1024.t7'))

st_vecs = {'train': st_vecs_train, 'val': st_vecs_val, 'test': st_vecs_test}
stid2idx = {'train': {}, 'val': {}, 'test': {}}

for part in ['train', 'val', 'test']:
    for i, id in enumerate(st_vecs[part]['ids']):
        stid2idx[part][id] = i

print("Done loading instruction vectors.", time.time() - t)

# print(stid2idx)
# exit()

print('Loading dataset.')
# print DATASET
print(DATASET)

print('Loading ingr vocab.')
with open(opts.vocab) as f_vocab:
    ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
    ingr_vocab['</i>'] = 1

with open('../data/recipe1M/classes_foodcom.pkl', 'rb') as f:
    class_dict = pickle.load(f)
    # id2class label (9: 8 + 1)
    id2class = pickle.load(f)
    print(len(class_dict), id2class)

with open('../data/recipe1M/triplet_sample.txt') as sample:
    sample_ids = []
    sample = sample.read().strip().split('\n')
    for item in sample:
        id = item.split('\t')[0]
        if id not in sample_ids:
            sample_ids.append(id)

st_ptr = 0
numfailed = 0

# if exist, remove the directory
if os.path.isdir('../data/train_foodcom_sample_lmdb'):
    shutil.rmtree('../data/train_foodcom_sample_lmdb')
if os.path.isdir('../data/val_foodcom_sample_lmdb'):
    shutil.rmtree('../data/val_foodcom_sample_lmdb')
if os.path.isdir('../data/test_foodcom_sample_lmdb'):
    shutil.rmtree('../data/test_foodcom_sample_lmdb')

env = {'train': [], 'val': [], 'test': []}
env['train'] = lmdb.open('../data/train_foodcom_sample_lmdb', map_size=int(1e11))
env['val']   = lmdb.open('../data/val_foodcom_sample_lmdb', map_size=int(1e11))
env['test']  = lmdb.open('../data/test_foodcom_sample_lmdb', map_size=int(1e11))

print('Assembling dataset.')
img_ids = dict()
keys = {'train': [], 'val': [], 'test': []}

maxlength = 0

with open('foodcom_dataset.json') as json_file:
    dataset = json.load(json_file)

id_count = 0
for i, entry in tqdm(enumerate(dataset)):
    if entry['id'] in sample_ids:
        id_count += 1
        if id_count % 1000 == 0:
            print('processed {} sample data'.format(i))
        ninstrs = len(entry['instructions'])
        # print(ninstrs, i, entry.keys(), entry)

        ingr_detections = detect_ingrs(entry, ingr_vocab)
        ningrs = len(ingr_detections)
        # print('ingr_detections', ingr_detections)

        maxlen_temp = max(ninstrs, ningrs)
        if maxlen_temp > maxlength:
            maxlength = maxlen_temp

        # clean data::: filter recipe have larger than 20 instructions steps / > 20 ingredients
        # if ninstrs >= opts.maxlen or ningrs >= opts.maxlen or ningrs == 0 or remove_ids.get(entry['id']):
        #     continue

        ingr_vec = np.zeros((opts.maxlen), dtype='uint16')
        ingr_vec[:ningrs] = ingr_detections
        # print('ingr_vec', ingr_vec)

        partition = entry['partition']

        stpos = stid2idx[partition][entry['id']] # select the sample corresponding to the index in the skip-thoughts data
        beg = st_vecs[partition]['rbps'][stpos] - 1 # minus 1 because it was saved in lua

        # length of instructions
        end = beg + st_vecs[partition]['rlens'][stpos]

        # print(partition, 'stpos', stpos, 'beg', beg, 'end', end)

        serialized_sample = pickle.dumps({'ingrs': ingr_vec, \
                                           'intrs': st_vecs[partition]['encs'][beg:end], \
                                           'classes': class_dict[entry['id']] + 1}
                                          )
        with env[partition].begin(write=True) as txn:
            txn.put('{}'.format(entry['id']).encode('latin1'), serialized_sample)
        # keys to be saved in a pickle file
        keys[partition].append(entry['id'])
        # exit()

print('maxlength', maxlength)
# exit()
for k in keys.keys():
    with open('../data/{}_foodcom_sample_keys.pkl'.format(k), 'wb') as f:
        pickle.dump(keys[k], f)

'''
Assembling dataset < 20
1029720it [4:06:20, 69.67it/s]
Training samples: 238408 - Validation samples: 51119 - Testing samples: 51304

### get foodcom sample data
507834it [44:47, 188.93it/s]
maxlength 87
Training samples: 29545 - Validation samples: 6408 - Testing samples: 6407

### get foodcom data
506998it [5:32:49, 28.69it/s]processed 507000 data
507834it [5:33:21, 25.39it/s]
maxlength 143
Training samples: 355077 - Validation samples: 76463 - Testing samples: 76294

'''
print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))

