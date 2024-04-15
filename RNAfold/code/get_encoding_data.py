
'''
    Used to obtain npz data packages corresponding to the training set and test set through the encoding model.

'''
import os
import torch
import numpy as np
import argparse
from function import constraint_matrix_CDP, constraint_matrix_batch, travel
import sys
sys.path.insert(0, '../../util')
sys.path.insert(0, '../../ts2vec')

from utils import get_bpseq_table, get_ct_table, get_data, get_args_from_json
from ts2vec import TS2Vec


def get_encoding_data(file_type, data_path, file_num, repr_dim, name, limit_length, Save_dir, whether_pack_train, whether_pack_test):

    model = TS2Vec(
        input_dims=4,
        output_dims=repr_dim
    )

    model.load('../../ts2vec/model_file/{}.pth'.format(name))
    print('model load success')

    for i in range(len(file_num)):
        path = os.path.join(data_path, file_num[i])

        if file_num[i] == name.split('_')[0] + '-{}'.format(limit_length) and whether_pack_train == True: 
            
            print('get train data ...')
            if file_type == 'bpseq':
                train_seq_list, train_label_matrix, train_length, file_name = get_bpseq_table(path, limit_length)
            elif file_type == 'ct':
                train_seq_list, train_label_matrix, train_length, file_name = get_ct_table(path, limit_length)
            
            file_name = np.array(file_name)
            train_seq_list = get_data(np.array(train_seq_list))
            train_label_matrix = np.array(train_label_matrix, dtype=int)
            train_length = np.array(train_length, dtype=int)
            data_repr_train = model.encode(train_seq_list)
            train_seq_list = travel(train_seq_list)

            print('start get constraint matrix N ...')
            N = constraint_matrix_CDP(train_seq_list)
            print('get constraint N success !')
            print('start get constraint matrix M ...')
            M = constraint_matrix_batch(train_seq_list) # * N
            # M = N
            print('get constraint M success !')

            print('train_data: ', train_seq_list.shape, train_label_matrix.shape, data_repr_train.shape, N.shape, M.shape, train_length.shape, file_name.shape)
            np.savez(os.path.join(Save_dir, file_num[i]), train_seq_list, train_label_matrix, data_repr_train, N, M, train_length, file_name)
            print('train data get success !')
        
        elif file_num[i] == name.split('_')[1] and whether_pack_test == True:
            
            print('get test data ...')
            if file_type == 'bpseq':
                test_seq_list, test_label_matrix, test_length, file_name = get_bpseq_table(path, limit_length)
            elif file_type == 'ct':
                test_seq_list, test_label_matrix, test_length, file_name = get_ct_table(path, limit_length)

            file_name = np.array(file_name)
            test_seq_list = get_data(np.array(test_seq_list))
            test_label_matrix = np.array(test_label_matrix, dtype=int)
            test_length = np.array(test_length, dtype=int)
            data_repr_test = model.encode(test_seq_list)
            test_seq_list = travel(test_seq_list)
            print('start get constraint matrix N ...')
            N = constraint_matrix_CDP(test_seq_list)
            print('get constraint N success !')
            print('start get constraint matrix M ...')
            M = constraint_matrix_batch(test_seq_list) # * N
            # M = N
            print('get constraint M success !')
            sys.stdout.flush()
            print('test data: ', test_seq_list.shape, test_label_matrix.shape, data_repr_test.shape, N.shape, M.shape, test_length.shape, file_name.shape)
            np.savez(os.path.join(Save_dir, file_num[i] ), test_seq_list, test_label_matrix, data_repr_test, N, M, test_length, file_name)
            print('test data get success !')

    print('finish !')


def main():

    args = get_args_from_json('../../util/args_Arc128.json')
    # args = get_args_from_json('../../util/args_Arc512.json')
    limit_length = args.max_seq_len 
    file_type = args.file_type
    encoding_model_name = args.encoding_model_name + '-{}'.format(limit_length)
    repr_dim = args.repr_dim
    whether_pack_train = args.whether_pack_train
    whether_pack_test = args.whether_pack_test

    data_path = '../../ts2vec/datasets/{}/'.format(encoding_model_name)
    file_num = os.listdir(data_path) 
    Save_dir = "../data/{}".format(limit_length) 

    get_encoding_data(file_type, data_path, file_num, repr_dim, encoding_model_name, limit_length, Save_dir, whether_pack_train, whether_pack_test)


if __name__ == '__main__':
    
    main()
