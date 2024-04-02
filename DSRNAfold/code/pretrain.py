'''
    Used to obtain pre-trained models
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from function import plot_losses
from Layers import PretrainModel, CombinedLoss_BCE_Dice
import sys
sys.path.insert(0, '../../util')
from utils import get_args_from_json, set_random_seed


def pretrain(label_matrix, batch_size, epochs, feature_vector, N, limit_length, pretrain_model_name, device_num, lr, dropout_rate, pos_weight, save_loss_for_pretrain, accum_steps):
    
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    print('Device number: ',  device)
    dropout_rate = dropout_rate
    model = PretrainModel(dropout_rate=dropout_rate).to(device)
    model.train()
    print("Model Pretraining ... ")
    print('The total number of samples pretrained: ', label_matrix.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.90, 0.98), eps=1e-7)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    pos_weight = torch.Tensor([pos_weight]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    criterion = CombinedLoss_BCE_Dice(pos_weight = pos_weight)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=3, verbose=True, eps=4e-5, min_lr=1e-11)


    feature_vector, N, label_matrix = torch.from_numpy(feature_vector), torch.from_numpy(N), torch.from_numpy(label_matrix)

    dataset = TensorDataset(feature_vector, N, label_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    min_loss = float('inf')

    for epoch in range(epochs):
        loss = 0.0
        optimizer.zero_grad()
        for idx, data in enumerate(dataloader):
            inputs, inputs1, labels = data
            inputs, inputs1 = inputs.float(), inputs1.float()
            inputs, inputs1, labels = inputs.to(device), inputs1.to(device), labels.to(device)
            outputs = model(inputs, inputs1)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

        losses.append(loss.item())
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), f'../model/{limit_length}/{pretrain_model_name}_best.pt')
    
        scheduler.step(loss)
        
        plot_losses(losses, save_loss_for_pretrain, 0)

        print('Pretraining log: epoch: {}, loss: {}'.format(epoch, loss.item()))
        # if epoch % 10 == 0:  # Save every 10 epochs
        #    torch.save(model.state_dict(), f'../model/{limit_length}/{name}_{epoch}.pt')

    print('Finished Pretraining')
    torch.save(model.state_dict(), f'../model/{limit_length}/{pretrain_model_name}.pt')
    print('model save success !')


def main():

    # get args
    args = get_args_from_json('../../util/args_Arc128.json')
    
    device = args.device
    torch.cuda.set_device(device)
    # if gpu is to be used
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(args.seed)
    set_random_seed(args.seed)
    # get data
    limit_length = args.max_seq_len
    batch_size = args.batch_size
    epochs = args.pertrain_epochs
    dropout_rate = args.dropout_rate
    pos_weight = args.pos_weight
    family = args.pretrain_data
    lr = args.lr
    pretrain_model_name = args.pretrain_model_name
    save_loss_for_pretrain = args.save_loss_for_pretrain
    accum_steps = args.accum_steps

    family_ = '{}-{}.npz'.format(family, limit_length)
    Save_dir = "../data/{}".format(limit_length)
    data = np.load(os.path.join(Save_dir, family_))

    label_matrix, feature_vector, N = data['arr_1'], data['arr_2'], data['arr_3']

    # small samples for test
    # label_matrix, feature_vector = label_matrix[:500], feature_vector[:500]
    print(label_matrix.shape, feature_vector.shape, N.shape)

    pretrain(label_matrix, batch_size, epochs, feature_vector, N, limit_length, pretrain_model_name, device, lr, dropout_rate, pos_weight, save_loss_for_pretrain, accum_steps)


if __name__ == '__main__':

    main()

