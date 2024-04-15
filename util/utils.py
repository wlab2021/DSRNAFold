import os
import numpy as np
import pandas as pd
import pickle
import torch
import random
from datetime import datetime
import argparse
import json

def take_per_row(A, indx, num_elem):

    indx = indx[:,None]
    r = np.arange(num_elem)
    all_indx = indx + r

    row = torch.arange(all_indx.shape[0])[:,None]
    col = all_indx

    res = A[row, col]
    return res

def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr

def cosine_similarity(sim, z):
    
    norm_1 = torch.norm(z, p=2, dim=2)[:,:,None] # T * 2B * 1
    norm_2 = norm_1.transpose(1,2) # T * 1 * 2B
    norm = torch.matmul(norm_1, norm_2) # T * 2B * 2B
    sim = torch.div(sim, norm)
    return sim

def get_true_len(x):
    seq_num = x.size(0)  
    time_num = x.size(1) 
    min_seq_len = 100000
    for i in range(seq_num):
        true_seq_len = 0
        for j in range(time_num):
            if(x[i][j][0] == 0 and x[i][j - 1][0] != 0):
                true_seq_len = j
        min_seq_len = min(min_seq_len, true_seq_len)

    return min_seq_len

def get_data(data):
    res = np.zeros((data.shape[0], data.shape[1], len(data[0][0])))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(len(data[0][0])):
                res[i][j][k] = int(data[i][j][k])
    
    return np.array(res, dtype=int)

def code_padding(seq, seq_len):
    '''
        word_coder = {
        "A": 1,
        "U": 2,
        "G": 3,
        "C": 4,
    }
    '''
    word_coder = {
        "A": '0001',
        "U": '0010',
        "G": '0100',
        "C": '1000',
        "a": '0001',
        "u": '0010',
        "g": '0100',
        "c": '1000',
    }
    seq_ = []
    for i in range(len(seq), seq_len):
        seq.append('N')  
    for i in range(seq_len):
        feature = word_coder.get(seq[i], '0000')
        seq_.append(feature)
      
    return seq_
    
def get_ct_label(original_list, limit_length):
    seq = []

    label = np.zeros((limit_length, limit_length), dtype=int).tolist()
    for ele in original_list:

        if(ele[4] > 0):
            x, y = ele[0]-1, ele[4]-1
            label[x][y] = 1
        seq.append(ele[1])

    seq = code_padding(seq, limit_length)

    return seq, label, len(seq)


def get_ct_table(path, limit_length):

    seq_list = []
    label_matrix = []
    length = []

    f_name = os.listdir(path)

    for i in range(len(f_name)):

        ele = pd.read_csv(os.path.join(path, f_name[i]), sep="\s+", skiprows=1, header=None)
        ele = ele.values.tolist()
        if len(ele) <= limit_length:
            _seq, _label, _length = get_ct_label(ele, limit_length)
            seq_list.append(_seq)
            label_matrix.append(_label)
            length.append(_length)

    return seq_list, label_matrix, length, f_name


def get_bpseq_label(text, limit_length):
    seq = []

    label = [[0 for _ in range(limit_length)] for _ in range(limit_length)]
    length = len(text)
    for i in range(length):
        try:
            tx = text[i].split(' ')
            tx[2] = tx[2].split('\n')[0]
            seq.append(tx[1])
        except:
            tx = text[i].split('\t')
            tx[2] = tx[2].split('\n')[0]
            seq.append(tx[1])
        if int(tx[2]) > 0:
            x = int(tx[0]) - 1
            y = int(tx[2]) - 1
            label[x][y] = 1

    seq = code_padding(seq, limit_length)
    return seq, label, length


def get_bpseq_table(path, limit_length):
    seq_list = []
    label_matrix = []
    length = []
    f_name = os.listdir(path)

    for i in range(len(f_name)):
        with open(os.path.join(path, f_name[i]), 'r') as f:
            text = f.readlines()
            seq_, label_, length_ = get_bpseq_label(text, limit_length)
            seq_list.append(seq_)
            label_matrix.append(label_)
            length.append(length_)
            f.close()

    return seq_list, label_matrix, length, f_name

def get_seq_table(path, limit_length):
    seq_list = []
    label_matrix = []
    length = []
    f_name = os.listdir(path)

    for i in range(len(f_name)):
        with open(os.path.join(path, f_name[i]), 'r') as f:
            text = f.readlines()
            try:
                seq_, label_, length_ = get_bpseq_label(text, limit_length)
                seq_list.append(seq_)
                label_matrix.append(label_)
                length.append(length_)
            except:
                print(f_name[i])

            f.close()

    return seq_list, label_matrix, length

def get_fasta_table(path, limit_length):
    seq = []
    with open(path, 'r') as f:
        text = f.readlines()
        for i in range(1, len(text), 2):
            seq_ = text[i][:-1]
            seq_ = [char for char in seq_]
            seq.append(code_padding(seq_, limit_length))
            
        f.close()

    return seq


def set_random_seed(seed_value=0):
    
    random.seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_data', type=str, help='The path of data for pretrain')
    parser.add_argument('--train_data', type=str, help='The path of data for pretrain train')
    parser.add_argument('--test_data', type=str, help='The path of data for test')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='The dropout rate for pretrain model')
    parser.add_argument('--max_seq_len', type=int, choices=[128, 512], help='RNA sequence length')
    parser.add_argument('--device', type=int, default=0, help='The gpu no. used for training and predict')
    parser.add_argument('--batch-size', type=int, default=16, help='The batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--pertrain_epochs', type=int, default=120, help='The number of pretrain_epochs')
    parser.add_argument('--train_epochs', type=int, default=20, help='The number of train_epochs')
    parser.add_argument('--seed', type=int, default=10, help='The random seed')
    parser.add_argument('--pos_weight', type=float, default=300, help='The weight of positive samples in BCEloss function')
    parser.add_argument('--pretrain_model_name', type=str, help='The name of pretrain model')
    parser.add_argument('--constraint_steps', type=int, default=16, help='The num of constraint epochs')
    parser.add_argument('--save_train_model_dir', type=str, help='The dir name of train model')
    parser.add_argument('--start_test_model_id', type=int, help='The start id of test model')
    parser.add_argument('--end_test_model_id', type=int, help='The end id of test model')
    parser.add_argument('--get_bpseq_file', type=bool, default=False, help='Whether to get bpseq file')
    parser.add_argument('--normalize_by_threshold', type=float, default=0.5, help='Normalize the prediction by threshold')
    parser.add_argument('file_type', type=str, choices=['ct', 'bpseq'], help='The type of file')
    parser.add_argument('encoding_model_name', type=str, help='The name of encoding model')
    parser.add_argument('repr_dim', type=int, default=320, help='The dimension of representation')
    parser.add_argument('cut_min_len', type=int, default=20, help='The min length of cut for encoding model')
    parser.add_argument('cut_maxlen', type=int, default=70, help='The max length of cut for encoding model')
    parser.add_argument('encoding_model_epochs', type=int, default=10, help='The epochs of encoding model')
    parser.add_argument('--whether_pack_train', type=bool, help='Whether to pack train data in LTP_get_data')
    parser.add_argument('--whether_pack_test', type=bool, help='Whether to pack test data in LTP_get_data')
    parser.add_argument('--save_loss_for_pretrain', type=str, help='the path for save pretrain loss png')
    parser.add_argument('--save_loss_for_train', type=str, help='the path for save train loss png')
    parser.add_argument('--file_path', type=str, help='the primary path of data file')

    return parser.parse_args()


def get_args_from_json(json_file_path):
    
    with open(json_file_path, 'r') as f:
        params = json.load(f)
        f.close()
    
    args = argparse.Namespace(**params)

    return args