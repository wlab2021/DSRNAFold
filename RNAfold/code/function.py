'''
    Where creatmat_mine is derived from the source code of CDPfold and has been modified based on it.
'''
import os
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags
import matplotlib.pyplot as plt
import pandas as pd

def to_categorical_torch(x, num_classes):

    # Convert x to a long tensor
    x_tensor = torch.tensor(x).long() 
    one_hot = torch.nn.functional.one_hot(x_tensor, num_classes=num_classes)
    # Convert PyTorch tensor to numpy array
    
    return one_hot.numpy()

def constraint_matrix_batch(x):

    c = to_categorical_torch(x, num_classes=5)

    base_a = c[:, :, 1]
    base_u = c[:, :, 2]
    base_g = c[:, :, 3]
    base_c = c[:, :, 4]

    au = np.matmul(np.expand_dims(base_a, -1), np.expand_dims(base_u, 1))
    au_ua = au + np.transpose(au, (0, 2, 1))
    cg = np.matmul(np.expand_dims(base_c, -1), np.expand_dims(base_g, 1))
    cg_gc = cg + np.transpose(cg, (0, 2, 1))
    ug = np.matmul(np.expand_dims(base_u, -1), np.expand_dims(base_g, 1))
    ug_gu = ug + np.transpose(ug, (0, 2, 1))
    M = au_ua + cg_gc + ug_gu

    mask = diags([1] * 7, [-3, -2, -1, 0, 1, 2, 3], shape=(M.shape[-2], M.shape[-1])).toarray()
    mask = 1-mask
    M = M * mask

    return M

def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x, y):
    if x == 1 and y == 2:
        return 2
    elif x == 3 and y == 4:
        return 3
    elif x == 3 and y == 2:
        return 0.8
    elif x == 2 and y == 1:
        return 2
    elif x == 4 and y == 3:
        return 3
    elif x == 2 and y == 3:
        return 0.8
    else:
        return 0

def compute_pairing_scores(data):
    score_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            score_matrix[i, j] = paired(data[i], data[j])
    return score_matrix

def update_coefficient(coefficient, i, j, data, score_matrix, search_range, unpair_flag):
    for add in range(1, search_range):
        index1, index2 = i - add, j + add
        if 0 <= index1 < len(data) and 0 <= index2 < len(data):
            score = score_matrix[index1, index2]
            if score == 0:
                if unpair_flag == 1:
                    break
                unpair_flag = 1
            else:
                coefficient += score * Gaussian(add)
        else:
            break
    return coefficient

def create_mymat(data, search_range=30):
    len_data = len(data)
    mat = np.zeros((len_data, len_data))
    score_matrix = compute_pairing_scores(data)

    for i in range(len_data):
        for j in range(len_data):
            if abs(j - i) > 3:
                score_ij = score_matrix[i, j]
                coefficient = 2 * score_ij
                if score_ij > 0.0:
                    unpair_flag = 0
                else:
                    unpair_flag = 1
                coefficient = update_coefficient(coefficient, i, j, data, score_matrix, search_range, unpair_flag)
                coefficient = update_coefficient(coefficient, j, i, data, score_matrix, search_range, unpair_flag)
                mat[i, j] = coefficient / 2.0

    return mat

def creatmat_mine(data):
    mat = np.zeros([len(data), len(data)])
    search_range = 30
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            if abs(j - i) > 3:
                score_ij = paired(data[i], data[j])
                coefficient = 2 * score_ij
                if score_ij > 0.0:
                    unpair_flag = 0
                    for add in range(1, search_range):
                        if i - add >= 0 and j + add < len(data):
                            score = paired(data[i - add], data[j + add])
                            if score == 0 and unpair_flag == 0:
                                unpair_flag = 1
                            elif score == 0 and unpair_flag == 1:
                                break
                            else:
                                coefficient = coefficient + score * Gaussian(add)
                        else:
                            break
                    unpair_flag = 0
                    for add in range(1, search_range):
                        if i + add < len(data) and j - add >= 0:
                            score = paired(data[i + add], data[j - add])
                            if score == 0 and unpair_flag == 0:
                                unpair_flag = 1
                            elif score == 0 and unpair_flag == 1:
                                break
                            else:
                                coefficient = coefficient + score * Gaussian(add)
                        else:
                            break
                else:
                    for add in range(1, search_range):
                        if i - add >= 0 and j + add < len(data):
                            score = paired(data[i - add], data[j + add])
                            if score == 0:
                                break
                            else:
                                coefficient = coefficient + score * Gaussian(add)
                        else:
                            break
                    for add in range(1, search_range):
                        if i + add < len(data) and j - add >= 0:
                            score = paired(data[i + add], data[j - add])
                            if score == 0:
                                break
                            else:
                                coefficient = coefficient + score * Gaussian(add)
                        else:
                            break
            mat[[i], [j]] = coefficient / 2.0
    return mat

def constraint_matrix_CDP(x):
    N = []
    bz = x.shape[0]
    for i in range(bz):

        matrix = create_mymat(x[i])
        matrix = np.array(matrix)
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val > 0:
            normalized_matrix = (matrix - min_val) / (max_val - min_val)
            N.append(normalized_matrix)
        else:
            N.append(matrix)


    return np.array(N)

def F1_loss(y_pred, y_true):
    TP = torch.sum(y_true * y_pred)
    FP = torch.sum((1 - y_true) * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    F1  = 2 * TP / (2 * TP + FP + FN + 1e-8)
    loss = 1 - F1
    return loss

def calculate_for_contrast(y_true, y_pred, test_length):

    TP, FP, TN, FN = 0, 0, 0, 0

    label = y_true[0:test_length][0:test_length]
    prediction = y_pred[0:test_length][0:test_length]
    
    TP = torch.sum(label * prediction).item()
    FP = torch.sum((1 - label) * prediction).item()
    TN = torch.sum((1 - label) * (1 - prediction)).item()
    FN = torch.sum(label * (1 - prediction)).item()
    
    if TP + FP == 0:
        P = 0
    else:
        P = round(TP / (TP + FP), 4)
    if TP + FN == 0:
        R = 0
    else:
        R = round(TP / (TP + FN), 4)
    if P + R == 0:
        F1 = 0
    else:
        F1 = round(2 * P * R / (P + R), 4)
    
    return P, R, F1, TP, FP, TN, FN

def pre_rec_F1_contrast(y_true, y_pred, test_length, test_family, limit_length):

    all_results = pd.DataFrame()
    df = pd.DataFrame(columns=['P', 'R', 'F1'])  # Create a new DataFrame for each model
    for i in range(test_length.shape[0]):

        P, R, F1, TP, FP, TN, FN = calculate_for_contrast(y_true[i], y_pred[i], test_length[i])
        df = df.append({'P': P, 'R': R, 'F1': F1}, ignore_index=True)  # Append to DataFrame

    all_results = pd.concat([all_results, df], axis=1)  # Concatenate the results of each model horizontally

    all_results.to_excel('..\\TSLTP_{}-{}.xlsx'.format(test_family, limit_length), index=False)  # Save the final DataFrame to Excel

    print('get excel result successfully !')

def calculate(y_true, y_pred, test_length):

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(test_length.shape[0]):
        label = y_true[i][0:test_length[i]][0:test_length[i]]
        prediction = y_pred[i][0:test_length[i]][0:test_length[i]]
        tp = torch.sum(label * prediction).item()
        fp = torch.sum((1 - label) * prediction).item()
        tn = torch.sum((1 - label) * (1 - prediction)).item()
        fn = torch.sum(label * (1 - prediction)).item()
        
        TP += tp
        FP += fp
        TN += tn
        FN += fn

    return TP, FP, TN, FN

def pre_rec_F1(y_true, y_pred, test_length):

    TP, FP, TN, FN = calculate(y_true, y_pred, test_length)
    if TP + FP == 0:
        precision = 0
    else:
        precision = round(TP / (TP + FP), 3)
    if TP + FN == 0:
        recall = 0
    else:
        recall = round(TP / (TP + FN), 3)
    
    F1 = round(2 * TP / (2 * TP + FP + FN), 3)

    return precision, recall, F1

def find_last_no_zero(seq):
    index = -1
    for i in range(len(seq)):
        if seq[i] != 0:
            index = i
    return index + 1

def num_to_letter(seq):
    length = find_last_no_zero(seq)
    for i in range(length):
        if(seq[i] == 1):
            seq[i] = 'A'
        elif(seq[i] == 2):
            seq[i] = 'U'
        elif(seq[i] == 3):
            seq[i] = 'G'
        elif (seq[i] == 4):
            seq[i] = 'C'
        else:
            seq[i] = 'N'
    return seq[:length]

def travel(x):
    m = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):  
        for j in range(x.shape[1]):  
            if x[i][j][-1] == 1:  # A
                m[i][j] = 1
            elif x[i][j][-2] == 1:  # U
                m[i][j] = 2
            elif x[i][j][-3] == 1:  # G
                m[i][j] = 3
            elif x[i][j][-4] == 1:  # C
                m[i][j] = 4
    return m

def normalize_by_threshold(prediction, threshold):
    prediction[prediction < threshold] = 0
    prediction[prediction >= threshold] = 1
    return prediction

def postprocess(prediction):
    cnt = np.zeros([len(prediction), len(prediction[0])])
    for k in range(len(prediction)):  
        x = prediction[k]
        for i in range(len(x)): 
            for j in range(len(x[0])):
                if(x[i][j] == 1):
                    cnt[k][i] += 1
        prediction[k] = x
 
    for k in range(len(prediction)): 
        x = prediction[k]
        for i in range(len(x)):
            if(cnt[k][i] > 1):
                for j in range(len(x[0])):
                    if(cnt[k][i] > 1 and x[i][j] == 1):
                        x[i][j] = 0
                        x[j][i] = 0
                        cnt[k][i] -= 1
                        cnt[k][j] -= 1
                for j in range(len(x[0])):
                    if(cnt[k][i] == 1 and x[i][j] == 1):
                        for m in range(len(x[0])):
                            if(x[j][m] == 1 and m != i):
                                x[j][m] = 0
                                x[m][j] = 0
                                cnt[k][j] -= 1
                                cnt[k][m] -= 1
        prediction[k] = x

    return prediction
 
def is_symmetry(matrix):
    flag = True
    for k in range(len(matrix)):
        x = matrix[k]
        y = x.transpose()
        for i in range(len(x)):
            for j in range(len(x[0])):
                if(x[i][j] != y[i][j]):
                    flag = False
    return flag

def islegal(prediction):
    flag = True
    cnt = np.zeros([len(prediction), len(prediction[0])])  
    for k in range(len(prediction)):
        x = prediction[k]
        for i in range(len(x)): 
            for j in range(len(x[0])):
                cnt[k][i] += x[i][j]
            if(cnt[k][i] > 1):
                flag = False
                return flag
    return flag

def print_result(prediction, seq_list, label_atrix, test_length):
    res = 0
    letter = ['N', 'A', 'U', 'G', 'C']
    for k in range(len(prediction)):
        # length = find_last_no_zero(seq_list[k])
        length = test_length[k]
        flag = True
        for i in range(len(prediction[0])):
            cnt = 0
            for j in range(len(prediction[0][0])):
                cnt += prediction[k][i][j]
            if (cnt > 1):
                flag = False
        if (flag == False):
            res += 1
            with open('error/prediction_{}.txt'.format(k), 'w') as f:
                for h in range(length):
                    f.write(letter[seq_list[k][h]])
                f.write('\n')
                for i in range(length):
                    f.write(str(i + 1) + ' ')
                    for j in range(length):
                        if (prediction[k][i][j] != 0):
                            f.write(str(j + 1) + ' ')
                            f.write(str(prediction[k][i][j]) + ' ')
                    f.write('\n')
                f.close()

            with open('error/label_{}.txt'.format(k), 'w') as f:
                for h in range(length):
                    f.write(letter[seq_list[k][h]])
                f.write('\n')
                for i in range(length):
                    f.write(str(i + 1) + ' ')
                    for j in range(length):
                        if (label_atrix[k][i][j] == 1):
                            f.write(str(j + 1) + ' ')
                    f.write('\n')
                f.close()
    return res

def get_bpseq(test_length, seq_list, prediction, label_matrix, test_name):
   
    letter = ['N', 'A', 'U', 'G', 'C']
    
    for i in range(len(test_length)):
        with open('../result/predict/{}'.format(test_name[i]), 'w') as f:
            length = test_length[i]
            for h in range(length):
                f.write('{} '.format(h+1))
                f.write('{} '.format(letter[seq_list[i][h]]))
                flag = 0
                for k in range(len(prediction[i][h])):
                    if prediction[i][h][k] == 1:
                        flag = k + 1
                        break
                f.write('{}\n'.format(flag))
            f.close()

        with open('../result/label/{}'.format(test_name[i]), 'w') as f:
            length = test_length[i]
            for h in range(length):
                f.write('{} '.format(h+1))
                f.write('{} '.format(letter[seq_list[i][h]]))
                flag = 0
                for k in range(len(label_matrix[i][h])):
                    if label_matrix[i][h][k] == 1:
                        flag = k + 1
                        break
                f.write('{}\n'.format(flag))
            f.close()

def shuffle(seq_list, label_matrix, feature_vector, t):
    np.random.seed(t)
    order = np.random.permutation(seq_list.shape[0])
    seq_list = seq_list[order, :]
    label_matrix = label_matrix[order, :, :]
    feature_vector = feature_vector[order, :, :]

    return seq_list, label_matrix, feature_vector

def create_model_dir(path, name):

    path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
        return True

    return False

def plot_losses(losses, save_path, flag):
    
    plt.plot(losses)
    plt.xlabel('Epoch')
    if flag == 0:
        plt.ylabel('BCE Loss')
    if flag == 1:
        plt.ylabel('F1 Loss')

    plt.savefig(save_path)
    plt.close()

def get_bpseq_to_contrast(test_length, seq_list, prediction, test_name):
    letter = ['N', 'A', 'U', 'G', 'C']
    for i in range(len(test_length)):
        with open('../result/predict/{}'.format(test_name[i]), 'w') as f:
            length = test_length[i]
            for h in range(length):
                f.write('{} '.format(h+1))
                f.write('{} '.format(letter[seq_list[i][h]]))
                flag = 0
                for k in range(len(prediction[i][h])):
                    if prediction[i][h][k] == 1:
                        flag = k + 1
                        break
                f.write('{}\n'.format(flag))
            f.close()

def get_excel_result(y_true, y_pred, test_length):

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(test_length.shape[0]):
        
        label = y_true[i][0:test_length[i]][0:test_length[i]]
        prediction = y_pred[i][0:test_length[i]][0:test_length[i]]

        tp = torch.sum(label * prediction).item()
        fp = torch.sum((1 - label) * prediction).item()
        tn = torch.sum((1 - label) * (1 - prediction)).item()
        fn = torch.sum(label * (1 - prediction)).item()
        
        if TP + FP == 0:
            precision = 0
        else:
            precision = round(tp / (tp + fp), 3)
        if TP + FN == 0:
            recall = 0
        else:
            recall = round(tp / (tp + fn), 3)
        if precision + recall == 0:
            F1 = 0
        else:
            F1 = round(2 * tp / (2 * tp + fp + fn), 3)
        
    return TP, FP, TN, FN


def cal_CM_person(seq_list, prediction, test_length, limit_length):

    letter = ['N', 'A', 'U', 'G', 'C']
    
    for i in range(len(prediction)):

        seq = seq_list[i]    
        sequence = [letter[k] for k in seq]
        with open('../ECM-{}/{}.bpseq'.format(limit_length, i), 'w') as f:  
            for j in range(test_length[i]):
                f.write(str(j+1))
                f.write(' ')
                f.write(sequence[j])
                f.write(' ')
                pairs = -10
                for k in range(test_length[i]):
                    # pairs += prediction[i][j][k]
                    pairs = max(pairs, prediction[i][j][k])
                unpairs = 1 - pairs
                f.write(str(unpairs))
                f.write('\n')
