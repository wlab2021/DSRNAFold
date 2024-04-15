'''
    Used to obtain trained models
'''
import os
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Layers import PretrainModel, Model, DiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from function import create_model_dir, F1_loss, plot_losses
import sys
sys.path.insert(0, '../../util')
from utils import get_args_from_json, set_random_seed


def train(feature_vector, label_matrix, N, M, batch_size, epochs,
                limit_length, device, save_mode_file, steps, premodel,
          dropout_rate, lr, save_loss_for_train, accum_steps):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print('Device number: ', device)
    print("Model Training ... ")

    model_name = '../model/{}/{}.pt'.format(limit_length, premodel)
    print('The total number of training samples: ', label_matrix.shape[0])
    
    steps = steps
    pre_model = PretrainModel(dropout_rate)
    pre_model.load_state_dict(torch.load(model_name))
    model = Model(pre_model, limit_length, steps).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-7)
    criterion = F1_loss
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=3, verbose=True, eps=1e-4, min_lr=1e-11)

    feature_vector = torch.from_numpy(feature_vector).float()
    label_matrix = torch.from_numpy(label_matrix).float()
    N = torch.from_numpy(N).float()
    M = torch.from_numpy(M).float()

    train_dataset = TensorDataset(feature_vector, N, M, label_matrix)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print('start: training ...')
    losses = []
    for epoch in range(epochs):
        loss = 0.0
        optimizer.zero_grad()
        for idx, data in enumerate(train_loader):
            inputs, inputs1, M, labels = data
            inputs, inputs1 = inputs.float(), inputs1.float()
            inputs, inputs1, M, labels = inputs.to(device), inputs1.to(device), M.to(device), labels.to(device)

            outputs = model(inputs, inputs1, M)    
            loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=200, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step(loss)
        losses.append(loss.item())
        plot_losses(losses, save_loss_for_train, 1)

        print('Training log: epoch: %d, F1_loss: %f' % (epoch + 1, loss.item()))
        torch.save(model.state_dict(), '../model/{}/{}/{}_{}.pt'.format(limit_length, save_mode_file, save_mode_file, epoch + 1))

    print('Finished Training')
    # torch.save(model.state_dict(), '../model/{}/{}-{}.pt'.format(limit_length, save_mode_file, limit_length))
    # print('model save success')


def main():

    # get args
    # args = get_args_from_json('../../util/args_Arc128.json')
    args = get_args_from_json('../../util/args_Arc512.json')

    device = args.device
    torch.cuda.set_device(device)
    # if gpu is to be used
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(args.seed)
    set_random_seed(args.seed)
    # get data
    limit_length = args.max_seq_len
    batch_size = args.train_batch_size
    epochs = args.train_epochs
    train_family = args.train_data
    steps = args.constraint_steps
    premodel = args.pretrain_model_name
    save_mode_file = args.save_train_model_dir
    dropout_rate = args.dropout_rate
    # dropout_rate = 0.4
    lr = args.train_lr
    save_loss_for_train = args.save_loss_for_train
    accum_steps = args.train_accum_steps

    train_family_ = train_family + '-{}.npz'.format(limit_length)
    Save_dir = "../data/{}".format(limit_length)
    train_data = np.load(os.path.join(Save_dir, train_family_))

    train_label_matrix, train_feature_vector, N, M = train_data['arr_1'], train_data['arr_2'], train_data['arr_3'], train_data['arr_4']

    flag = create_model_dir(Save_dir.replace('data', 'model'), save_mode_file)
    if flag == True:
        print('create model_dir success !')
    else:
        print("model_dir exists !")
        return 0

    # small samples for test
    # train_label_matrix, train_feature_vector, M = train_label_matrix[:500], train_feature_vector[:500], M[:500]
    print(train_label_matrix.shape, train_feature_vector.shape, N.shape, M.shape)
    
    # train
    train(train_feature_vector, train_label_matrix, N, M, batch_size, epochs, limit_length, device, save_mode_file, 
          steps, premodel, dropout_rate, lr, save_loss_for_train, accum_steps)


if __name__ == '__main__':
    
    main()