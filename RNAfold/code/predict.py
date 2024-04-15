'''
 Used for model prediction
'''

import os
import numpy as np
import torch   
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from Layers import PretrainModel, Model
from function import postprocess, is_symmetry, islegal, print_result, normalize_by_threshold, get_bpseq, pre_rec_F1, get_bpseq_to_contrast, pre_rec_F1_contrast
import sys
sys.path.insert(0, '../../util')
from utils import get_args_from_json

def start_predict(model_name, seq_list, feature_vector, label_matrix, N, M, limit_length, test_length, batch_size, device, start_id, end_id, flag, 
                  steps, dropout_rate, normalize_threshold, get_bpseq_file, test_family, test_name):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print('Device number: ', device)
    print('The total number of samples used for prediction.: ',  int(seq_list.shape[0]))

    feature_vector = torch.from_numpy(feature_vector).float()
    N = torch.from_numpy(N).float()    
    M = torch.from_numpy(M).float()
    label_matrix = torch.from_numpy(label_matrix).float()

    print('test data message: ', label_matrix.shape, feature_vector.shape, N.shape, M.shape)

    test_dataset = TensorDataset(feature_vector, N, M)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print('start predict ...')

    for i in range(start_id, end_id+1 ):
            

            model_path = os.path.join(model_name, model_name.split('/')[-1] + '_{}.pt'.format(i))
            # print(model_path)
            model = Model(PretrainModel(dropout_rate=dropout_rate), limit_length, steps=steps)
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            model.eval()

            prediction = []
            # with torch.no_grad():
            for _, data in enumerate(test_loader):
                feature_vector, N, M = data
                feature_vector, N, M = feature_vector.to(device), N.to(device), M.to(device)
                predict = model(feature_vector, N, M)
                predict = predict.cpu().detach().numpy()
        
                predict = normalize_by_threshold(predict, normalize_threshold)
                predict = postprocess(predict)
                prediction.extend(predict)

            # print()
            # whether_symmetry = is_symmetry(prediction)
            # print('Is the result matrix a symmetric matrix', whether_symmetry)
            # T = islegal(prediction)
            # print('Are all pairs at most one-to-one:', T)
 
            # Print the predicted results and the labels
            # get_bpseq(test_length, seq_list, prediction, label_matrix)


            P, R, F1 = pre_rec_F1(label_matrix, prediction, test_length)
            print()
            print("model id: ", i)
            print('Precision: ', P)
            print('Recall: ', R)
            print('F1: ', F1)
            print()

            if get_bpseq_file:
                # pre_rec_F1_contrast(label_matrix, prediction, test_length, test_family, limit_length)
                # get_bpseq_to_contrast(test_length, seq_list, prediction, test_name)
                get_bpseq(test_length, seq_list, prediction, label_matrix, test_name)

def main():

    # get args from json file
    # args = get_args_from_json('../../util/args_Arc128.json')
    args = get_args_from_json('../../util/args_Arc512.json')
    # args = get_args_from_json('../../util/args_TS0128.json')

    torch.manual_seed(args.seed)
    limit_length = args.max_seq_len
    start_id, end_id = args.start_test_model_id, args.end_test_model_id
    model_dir = args.save_train_model_dir
    test_family = args.test_data
    add_CDP_constraint = args.if_add_CDP_constraint
    get_bpseq_file = args.get_bpseq_file
    steps = args.constraint_steps
    dropout_rate = args.dropout_rate
    normalize_threshold = args.normalize_by_threshold
    device = args.device
    batch_size = args.pre_batch_size

    torch.cuda.set_device(device)
    # if gpu is to be used
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")

    Save_dir = "../data/{}".format(limit_length)
    model_dir = os.path.join(Save_dir.replace('data', 'model'), model_dir)
    test_data = np.load(os.path.join(Save_dir, '{}-{}.npz'.format(test_family, limit_length)))
    test_seq_list, test_label_matrix, test_feature_vector, N, M, test_length, test_name = test_data['arr_0'], test_data['arr_1'], test_data['arr_2'], test_data['arr_3'], test_data['arr_4'], test_data['arr_5'], test_data['arr_6']
    torch.cuda.empty_cache()
    # small samples for test
    # test_seq_list, test_label_matrix, test_feature_vector, N, M, test_length, test_name = test_seq_list[:10], test_label_matrix[:10], test_feature_vector[:10], N[:10], M[:10], test_length[:10], test_name[:10]
    
    start_predict(model_dir, test_seq_list, test_feature_vector, test_label_matrix, N, M, limit_length, test_length, batch_size, device, start_id, end_id, 
                  add_CDP_constraint, steps, dropout_rate, normalize_threshold, get_bpseq_file, test_family, test_name)


if __name__ == '__main__':
    
    main()