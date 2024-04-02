import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), #输出为输入的2倍
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class PretrainModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        
        super(PretrainModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        #self.generator = Generator(1)  # You need to implement this
        self.U_Net = U_Net()

    def forward(self, feature_vector, N):
        
        # Dropout
        feature_vector = self.dropout(feature_vector)

        feature_pair = self.concatD(feature_vector)
        # 放入generator
        # outputs = self.generator(encode_pair, self.max_seq_len, self.model_dim, 1)
        
        # add: N
        N = N.unsqueeze(-1)
        N = N.permute(0, 3, 1, 2)
        feature_pair = torch.cat((feature_pair, N), dim=1)

        # 放入 U-Net
        outputs = self.U_Net(feature_pair)
        
        # outputs = outputs.squeeze(-1)
        # 对称化
        # outputs = (outputs + outputs.transpose(1, 2)) / 2
        
        return outputs


    def concatD(self, feature_vector):

        # 变换为二维矩阵
        even_feature = feature_vector[:, :, 0::2]
        odd_feature = feature_vector[:, :, 1::2]
        N = even_feature.shape[1]
        x_l = even_feature.unsqueeze(2).expand(-1, -1, N, -1)
        x_r = odd_feature.unsqueeze(1).expand(-1, N, -1, -1)
        feature_pair = torch.cat([x_l, x_r], dim=-1)
        
        feature_pair = feature_pair.permute(0, 3, 1, 2)
        
        return feature_pair

class U_Net(nn.Module):
    def __init__(self,img_ch=21,output_ch=1, CH_FOLD2=1):
        super(U_Net,self).__init__()

        #池化操作
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #五次Double conv
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(32*CH_FOLD2))  
        self.Conv2 = conv_block(ch_in=int(32*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Conv3 = conv_block(ch_in=int(64*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Conv4 = conv_block(ch_in=int(128*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Conv5 = conv_block(ch_in=int(256*CH_FOLD2),ch_out=int(512*CH_FOLD2))

        #四次Up conv和 Double conv
        self.Up5 = up_conv(ch_in=int(512*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))

        self.Up4 = up_conv(ch_in=int(256*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))
        
        self.Up3 = up_conv(ch_in=int(128*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))
        
        self.Up2 = up_conv(ch_in=int(64*CH_FOLD2),ch_out=int(32*CH_FOLD2))
        self.Up_conv2 = conv_block(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))

        #一次'全连接'
        self.Conv_1x1 = nn.Conv2d(int(32*CH_FOLD2),output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        #conv 1*1
        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)
        return torch.transpose(d1, -1, -2) * d1  # 行列互换

class Model(nn.Module):
    def __init__(self, pre_model, seq_len, steps):
    
        super(Model, self).__init__()
        self.pre_model = pre_model
        self.seq_len = seq_len
        self.steps = steps
        self.s = nn.Parameter(torch.tensor(0.5))
        self.w = nn.Parameter(torch.rand(1))
        self.rho_p = nn.Parameter(torch.rand(self.seq_len, self.seq_len))
        self.alpha = nn.Parameter(torch.tensor(0.005))
        self.belt = nn.Parameter(torch.tensor(0.05))
        self.lr_alpha = nn.Parameter(torch.tensor(0.99))
        self.lr_belt = nn.Parameter(torch.tensor(0.99))
    
    def T_A(self, inputs, M):
        return (inputs + inputs.transpose(1, 2)) / 2 * M

    def PPcell(self, U, M, Lm, A, A_hat, t):
        G = U / 2 - (Lm * torch.sign(torch.sum(A, dim=-1) - 1)).unsqueeze(2).repeat(1, 1, self.seq_len)
        A_hat = A_hat + self.alpha * torch.pow(self.lr_alpha, t) * A_hat * M * (G + G.transpose(1, 2))
        A_hat = F.relu(torch.abs(A_hat) - self.rho_p * self.alpha * torch.pow(self.lr_alpha, t))
        A_hat = 1 - F.relu(1 - A_hat)
        A = self.T_A(A_hat, M)
        Lm = Lm + self.belt * torch.pow(self.lr_belt, t) * F.relu(torch.sum(A, dim=-1) - 1)
        return Lm, A, A_hat

    def forward(self, x, x1, M):

        scores = self.pre_model(x, x1)
        U = scores - self.s
        A_hat = scores
        A = self.T_A(A_hat, M)
        Lm = self.w * F.relu(torch.sum(A, dim=-1) - 1)
        A_list = []
        for t in range(self.steps):
            Lm, A, A_hat = self.PPcell(U, M, Lm, A, A_hat, t)
            A_list.append(A)
        
        return A_list[-1]

class ConstraintAndResult(nn.Module):
    def __init__(self, seq_len, steps):
        super(ConstraintAndResult, self).__init__()
        
        self.seq_len = seq_len
        self.steps = steps
        self.s = nn.Parameter(torch.tensor(0.5))
        self.w = nn.Parameter(torch.rand(1))
        self.rho_p = nn.Parameter(torch.rand(self.seq_len, self.seq_len))
        self.alpha = nn.Parameter(torch.tensor(0.005))
        self.belt = nn.Parameter(torch.tensor(0.05))
        self.lr_alpha = nn.Parameter(torch.tensor(0.99))
        self.lr_belt = nn.Parameter(torch.tensor(0.99))

    def T_A(self, inputs, M):
        return (inputs + inputs.transpose(1, 2)) / 2 * M

    def PPcell(self, U, M, Lm, A, A_hat, t):
        G = U / 2 - (Lm * torch.sign(torch.sum(A, dim=-1) - 1)).unsqueeze(2).repeat(1, 1, self.seq_len)
        A_hat = A_hat + self.alpha * torch.pow(self.lr_alpha, t) * A_hat * M * (G + G.transpose(1, 2))
        A_hat = F.relu(torch.abs(A_hat) - self.rho_p * self.alpha * torch.pow(self.lr_alpha, t))
        A_hat = 1 - F.relu(1 - A_hat)
        A = self.T_A(A_hat, M)
        Lm = Lm + self.belt * torch.pow(self.lr_belt, t) * F.relu(torch.sum(A, dim=-1) - 1)
        return Lm, A, A_hat

    def forward(self, scores, M):
        U = scores - self.s
        A_hat = scores
        A = self.T_A(A_hat, M)
        Lm = self.w * F.relu(torch.sum(A, dim=-1) - 1)
        A_list = []
        for t in range(self.steps):
            Lm, A, A_hat = self.PPcell(U, M, Lm, A, A_hat, t)
            A_list.append(A)
        return A_list[-1]

class CombinedLoss_BCE_Dice(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, pos_weight=200):
        super(CombinedLoss_BCE_Dice, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(torch.device(0)))
        self.Dice_loss = DiceLoss()

    def forward(self, y_pred, y_true):
        loss_ce = self.bce_loss(y_pred, y_true)
        loss_dice = self.Dice_loss(y_pred, y_true)
        return self.weight_dice * loss_dice + self.weight_ce * loss_ce

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice


