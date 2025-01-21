from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import copy
from models import iTransformer, MLP, MLP_3L, Regressor
from thop import profile, clever_format
from prettytable import PrettyTable
from torch.optim import lr_scheduler
# from exp.ContrastiveLoss import SupConLoss,swavloss,NTXentLoss
from exp.Regularization  import Regularization
import json
import glob
import psutil
from layers.Autoformer_EncDec import series_decomp

# def group_lasso_loss(pred: torch.Tensor, target: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
#     """
#     Computes the Group Lasso (L21) loss between prediction and ground truth tensors.
    
#     Args:
#         pred (torch.Tensor): Prediction tensor of shape (C, T).
#         target (torch.Tensor): Ground truth tensor of shape (C, T).
#         lambda_ (float): Regularization coefficient for the Group Lasso loss.

#     Returns:
#         torch.Tensor: Computed Group Lasso loss (scalar).
#     """
#     # print(pred.shape, target.shape)
#     assert pred.shape == target.shape, "Shape mismatch between prediction and target tensors"

#     # Compute the element-wise difference (error)
#     diff = pred - target

#     # Compute the L2 norm for each group (row-wise)
#     l2_norms = torch.norm(diff, p=2, dim=2) / diff.shape[2]

#     # Sum the L2 norms to get the L21 loss
#     l21_loss = lambda_ * torch.sum(l2_norms, dim=1).mean()

#     return l21_loss

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    return top_list
    # top_list = top_list.detach().cpu().numpy()
    # period = x.shape[1] // top_list
    # return period, abs(xf).mean(-1)[:, top_list]
    

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d
        loss = F.smooth_l1_loss(d, t_d)
        return loss

dist_criterion = RkdDistance()

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss

angle_criterion = RKdAngle()

class AttentionTransfer(nn.Module):
    def forward(self, args, student, teacher):
        # s_attention = F.normalize(torch.matmul(student, student.transpose(1, 2)))
        s_attention = torch.matmul(student, student.transpose(1, 2))
        # s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            # t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))
            t_attention = teacher.mean(dim=1)
            t_attention = t_attention[:, :args.enc_in, :args.enc_in]
            # t_attention = F.normalize(t_attention)
            t_attention = t_attention

        # print(s_attention.shape, t_attention.shape)

        # return (s_attention - t_attention).pow(2).mean()
        return (s_attention - t_attention).mean()

attention_criterion = AttentionTransfer()

class multi_scale_process_inputs(nn.Module):
    def __init__(self, down_sampling_layers=3, down_sampling_window=2):
        super(multi_scale_process_inputs, self).__init__()
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window

    def forward(self, x_enc):
        down_pool = torch.nn.AvgPool1d(kernel_size=self.down_sampling_window)
        
        # B,T,C -> B,C,T
        # x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            
        x_enc = x_enc_sampling_list

        return x_enc

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        if args.distillation:
            args.model_t = args.model_t.split(' ')
            self.model_t = []
            for model_t_name in args.model_t:
                print(model_t_name)
                args_t = copy.deepcopy(args)
                with open('config.json', 'r') as f:
                    config = json.load(f)
                model_config = config[self.args.model_id.split('_')[0]][model_t_name]
                for key, value in model_config.items():
                    setattr(args_t, key, value)
                self.args_t = args_t
                model_t = self.model_dict[model_t_name].Model(args_t).float().to(self.device)
                print("path:", f"./checkpoints/long_term_forecast_{self.args.model_id}_{model_t_name}")
                checkpoint_path = glob.glob(f"./checkpoints/long_term_forecast_{self.args.model_id}_{model_t_name}*/checkpoint.pth")[0]
                print("teacher checkpoint_path:", checkpoint_path)
                model_t.load_state_dict(torch.load(checkpoint_path))
                model_t.eval()
                self.model_t.append(model_t)

            # self.regressor1 = Regressor.Model(args, args.d_model, args_t.feature_dim, 2).float().to(self.device)
            # self.regressor1 = Regressor.Model(args, args.d_model, args.d_model).float().to(self.device)

            self.regressor = Regressor.Model(args, args_t.d_model, args.d_model).float().to(self.device)
            self.regressor.train()

            # self.decompsition = series_decomp(args.moving_avg)

            # weights_tensor = (1 / (args.enc_in * args.enc_in)) * torch.rand(
            #     args.enc_in, args.enc_in, args.pred_len // 2 + 1, dtype=torch.cfloat, device=self.device
            # )
            # self.weights = nn.Parameter(weights_tensor)

            # self.reconstructor = Regressor.Model(args, args.d_model, args_t.d_model).float().to(self.device)
            # self.reconstructor.train()
            # self.regressor1 = nn.Linear(args.d_model, args.d_model).to(self.device)
            # self.regressor2 = nn.Linear(args_t.feature_dim, args.d_model).to(self.device)
            # self.regressor1.train()
            # self.regressor2.train()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        count_parameters(model)
        # inputs = torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in)
        # macs, params = profile(model, inputs=(inputs,), verbose=False)
        # macs = clever_format([macs], "%.3f")
        # print(f"MACs: {macs}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(list(self.model.parameters()) + list(self.regressor1.parameters()) + list(self.regressor2.parameters()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.distillation:
            model_optim = optim.Adam(list(self.model.parameters()) + list(self.regressor.parameters()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def select_regularization(self):
        self.weight_decay=self.args.cf_weight_decay
        #print("self.weight_decay=",self.weight_decay)
        if self.weight_decay>0:
            self.reg_loss=Regularization(self.model, self.args.cf_weight_decay, self.args.cf_p).to(self.device)
        #else:
            #print("no regularization")

    def train(self, setting):
        # if self.args.distillation:
        #     print(f"./checkpoints/long_term_forecast_{self.args.model_id}_{self.args.model}_{self.args.data}*/checkpoint.pth")
        #     checkpoint_path = glob.glob(f"./checkpoints/long_term_forecast_{self.args.model_id}_{self.args.model}_{self.args.data}*/checkpoint.pth")[0]
        #     print("student checkpoint_path:", checkpoint_path)
        #     self.model.load_state_dict(torch.load(checkpoint_path))
        #     self.model.train()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':
            train_steps = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        else:
            scheduler = None

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Attentive Imitation Loss (AIL)
        if self.args.ail:
            et_max, et_min = -1, np.inf
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, features = self.model_t(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                et_batch = np.mean(nn.MSELoss(reduction='none')(outputs, batch_y).cpu().detach().numpy(), axis=(1, 2))
                et_max = max(et_max, max(et_batch))
                et_min = min(et_min, min(et_batch))
            eta = et_max - et_min

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_grad = []
            train_loss_gt = []
            train_loss_feature = []
            train_loss_logit = []

            self.model.train()
            epoch_time = time.time()
            batch_times = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_time = time.time()
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # if self.args.gamma:
                #     batch_x.requires_grad = True
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.model == 'Fredformer':
                    self.select_regularization()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None

                    if self.args.model=='Pathformer':
                        outputs, balance_loss, features = self.model(batch_x)
                    elif self.args.model in ['iTransformer', 'CARD', 'TimeMixer']:
                        outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, features = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.model == 'CARD':
                        c = nn.L1Loss()
                        self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs *self.ratio
                        batch_y = batch_y *self.ratio
                        loss = c(outputs, batch_y)

                        use_h_loss = False
                        h_level_range = [4,8,16,24,48,96]
                        h_loss = None
                        if use_h_loss:
                            for h_level in h_level_range:
                                batch,length,channel = outputs.shape
                                # print(outputs.shape)
                                h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
                                h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
                                h_ratio = self.ratio[:h_outputs.shape[-2],:]
                                # print(h_outputs.shape,h_ratio.shape)
                                h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
                                h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)


                                h_outputs = h_outputs*h_ratio
                                h_batch_y = h_batch_y*h_ratio

                                h_ouputs_agg *= h_ratio
                                h_batch_y_agg *= h_ratio

                                if h_loss is None:
                                    h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                                else:
                                    h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                    elif self.args.distillation:
                        with torch.no_grad():
                            outputs_t, features_t = [], []
                            for model_t in self.model_t:
                                outputs_t_tmp, features_t_tmp = model_t(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                f_dim = -1 if self.args.features == 'MS' else 0
                                outputs_t_tmp = outputs_t_tmp[:, -self.args.pred_len:, f_dim:]
                                features_t_tmp = features_t_tmp[-2][:, :batch_x.shape[-1], f_dim:]
                                outputs_t.append(outputs_t_tmp)
                                features_t.append(features_t_tmp)

                        # loss_logit = criterion(outputs, outputs_t)
                        if self.args.alpha:
                            if self.args.ail:
                                phi = 1 - torch.mean(nn.MSELoss(reduction='none')(outputs_t, batch_y).detach(), dim=(1, 2)) / eta
                                loss_logit = self.args.alpha * torch.dot(phi, torch.mean(nn.MSELoss(reduction='none')(outputs, outputs_t), dim=(1, 2)))
                            else:
                                criterion_none = nn.MSELoss(reduction='none')
                                # loss_logit = self.args.alpha * criterion_none(outputs, batch_y)[criterion_none(outputs, batch_y) > criterion_none(outputs_t, batch_y)].mean()
                                if len(outputs_t) == 1:
                                    outputs_t = outputs_t[0]
                                    # loss_logit = self.args.alpha * criterion_none(outputs, batch_y)[criterion_none(outputs, batch_y) > criterion_none(outputs_t, batch_y)].mean()
                                    # loss_logit = self.args.alpha * criterion_none(outputs, outputs_t)[criterion_none(outputs, batch_y) > criterion_none(outputs_t, batch_y)].mean()
                                    
                                    # MSE matching
                                    # loss_logit = self.args.alpha * criterion(outputs, outputs_t)

                                    # lasso loss
                                    # loss_logit = self.args.alpha * group_lasso_loss(outputs, outputs_t)
                                    # loss_logit = self.args.alpha * group_lasso_loss(outputs.permute(0, 2, 1), outputs_t.permute(0, 2, 1))

                                    # frequence direct
                                    # outputs_t_ft = torch.fft.rfft(outputs_t, dim=1)
                                    # # frequency_list_t = abs(outputs_t_ft).mean(-1)
                                    # frequency_list_t = abs(outputs_t_ft)
                                    # frequency_list_t[:, 0, :] = 0
                                    # # _, top_list_t = torch.topk(frequency_list_t, frequency_list_t.shape[-1]//2)
                                    # outputs_ft = torch.fft.rfft(outputs, dim=1)
                                    # # frequency_list = abs(outputs_ft).mean(-1)
                                    # frequency_list = abs(outputs_ft)
                                    # frequency_list[:, 0, :] = 0
                                    # # batch_indices = torch.arange(frequency_list.size(0)).unsqueeze(1)
                                    # # loss_logit_frequency = self.args.alpha * criterion(frequency_list[batch_indices, top_list_t], frequency_list_t[batch_indices, top_list_t]) # frequence distillation
                                    # loss_logit_frequency = self.args.alpha * criterion(frequency_list, frequency_list_t) # frequence distillation

                                    # frequence distillation (KL)
                                    outputs_t_ft = torch.fft.rfft(outputs_t, dim=1)
                                    frequency_list_t = abs(outputs_t_ft)
                                    outputs_ft = torch.fft.rfft(outputs, dim=1)
                                    frequency_list = abs(outputs_ft)
                                    frequency_list_t = F.softmax(frequency_list_t[:, 1:, :] / 0.5, dim=1)
                                    frequency_list = F.softmax(frequency_list[:, 1:, :] / 0.5, dim=1)
                                    loss_logit_frequency = self.args.alpha * F.kl_div(
                                        torch.log(frequency_list + 1e-8),  
                                        frequency_list_t,           
                                        reduction='mean'                  
                                    )

                                    # multiscale matching
                                    outputs_multi_scale = multi_scale_process_inputs()(outputs.permute(0, 2, 1))
                                    outputs_t_multi_scale = multi_scale_process_inputs()(outputs_t.permute(0, 2, 1))
                                    num_outputs = len(outputs_multi_scale)
                                    loss_logit_multiscale = 0
                                    for i in range(num_outputs):
                                        loss_logit_multiscale += self.args.alpha * criterion(outputs_multi_scale[i], outputs_t_multi_scale[i])
                                    loss_logit_multiscale /= num_outputs

                                    loss_logit = loss_logit_frequency + loss_logit_multiscale
                                else:
                                    outputs_t_tensor = torch.stack(outputs_t)
                                    errors = (outputs_t_tensor - batch_y.unsqueeze(0)).pow(2).mean(dim=[1, 2, 3]).detach() 
                                    weights = F.softmax(1.0 / errors, dim=0)

                                    outputs_expanded = outputs.unsqueeze(0) 
                                    losses = criterion_none(outputs_expanded, outputs_t_tensor) 
                                    losses = losses.mean(dim=[1, 2, 3]) 

                                    loss_logit = self.args.alpha * torch.sum(weights * losses)

                                # loss_logit = self.args.alpha * criterion(outputs, outputs_t)
                                # loss_logit = self.args.alpha * (criterion(outputs, outputs_t) + criterion(outputs, outputs_t2))
                        else:
                            loss_logit = 0

                        # if self.args.beta:
                        #     # frequence distillation
                        #     outputs_ft = torch.fft.rfft(outputs.permute(0, 2, 1), norm='ortho', dim=-1)
                        #     outputs_t_ft = torch.fft.rfft(outputs_t.permute(0, 2, 1), norm='ortho', dim=-1)
                        #     outputs_ft = F.normalize(outputs_ft, dim=-1)
                        #     outputs_t_ft = F.normalize(outputs_t_ft, dim=-1)
                        #     outputs_ft = torch.einsum("bix,iox->box", outputs_ft, self.weights).abs()
                        #     outputs_t_ft = torch.einsum("bix,iox->box", outputs_t_ft, self.weights).abs()
                        #     outputs_ft = F.normalize(outputs_ft, dim=-1)
                        #     outputs_t_ft = F.normalize(outputs_t_ft, dim=-1)

                        #     loss_feature = self.args.beta * nn.L1Loss()(outputs_ft, outputs_t_ft) # frequence distillation

                        if self.args.beta:
                            if len(features_t) == 1:
                                features_t = features_t[0]
                                features_t = features_t.reshape(features_t.shape[0], features_t.shape[1], -1) # [32, 8, 325, 325]
                                features_t_reg = self.regressor(features_t)
                                # features_t_seasonal, feature_t_trend = self.decompsition(features_t_reg)
                                # features_t_recon = self.reconstructor(features_t_reg)

                            # loss_feature = self.args.beta * attention_criterion(self.args, features, features_t)

                            # mse distance
                            # features_t = self.regressor2(features_t)
                            # features_t, _ = self.regressor2(features_t.permute(0, 2, 1))
                            # features = self.mlp1(outputs.permute(0, 2, 1))
                            # features_t = self.mlp2(outputs_t.permute(0, 2, 1))
                            # print(features[-1].shape, features_t.shape)
                            # loss_feature = self.args.beta * criterion(features, features_t_reg)
                            # loss_feature = self.args.beta * criterion(features[-2], features_t_reg)

                            # frequence direct
                            # features_t_ft = torch.fft.rfft(features_t_reg.permute(0, 2, 1), dim=1)
                            # # frequency_list_t = abs(features_t_ft).mean(-1)
                            # frequency_list_t = abs(features_t_ft)
                            # frequency_list_t[:, 0, :] = 0
                            # # _, top_list_t = torch.topk(frequency_list_t, frequency_list_t.shape[-1]//2)
                            # features_ft = torch.fft.rfft(features[-2].permute(0, 2, 1), dim=1)
                            # # frequency_list = abs(features_ft).mean(-1)
                            # frequency_list = abs(features_ft)
                            # frequency_list[:, 0, :] = 0
                            # # batch_indices = torch.arange(frequency_list.size(0)).unsqueeze(1)
                            # # loss_logit_frequency = self.args.alpha * criterion(frequency_list[batch_indices, top_list_t], frequency_list_t[batch_indices, top_list_t]) # frequence distillation
                            # loss_feature_frequency = self.args.beta * criterion(frequency_list, frequency_list_t) # frequence distillation

                            # frequence distillation (KL)
                            features_t_ft = torch.fft.rfft(features_t_reg.permute(0, 2, 1), dim=1)
                            frequency_list_t = abs(features_t_ft)
                            features_ft = torch.fft.rfft(features[-2].permute(0, 2, 1), dim=1)
                            frequency_list = abs(features_ft)
                            frequency_list_t = F.softmax(frequency_list_t[:, 1:, :] / 0.5, dim=1)
                            frequency_list = F.softmax(frequency_list[:, 1:, :] / 0.5, dim=1)
                            loss_feature_frequency = self.args.beta * F.kl_div(
                                torch.log(frequency_list + 1e-8),  
                                frequency_list_t,           
                                reduction='mean'                  
                            )

                            # multi scale matching
                            features_multi_scale = multi_scale_process_inputs()(features[-2])
                            features_t_multi_scale = multi_scale_process_inputs()(features_t_reg)
                            loss_feature_multiscale = 0
                            num_features = len(features_multi_scale)
                            for i in range(num_features):
                                loss_feature_multiscale += self.args.beta * criterion(features_multi_scale[i], features_t_multi_scale[i])
                            loss_feature_multiscale /= num_features

                            # relation kd
                            features_r = features[-2].permute(1, 0, 2)
                            features_r = features_r.reshape(features_r.shape[0], -1)
                            features_t_r = features_t_reg.permute(1, 0, 2)
                            features_t_r = features_t_r.reshape(features_t_r.shape[0], -1)
                            loss_feature_relation = self.args.beta * dist_criterion(features_r, features_t_r)

                            loss_feature = loss_feature_frequency + loss_feature_multiscale + loss_feature_relation

                            # seasonal match
                            # featurs_seasonal, featurs_trend = self.decompsition(features[-2])
                            # loss_feature = self.args.beta * criterion(featurs_seasonal, features_t_seasonal)


                            # relation kd
                            # features = features.permute(1, 0, 2)
                            # features = features.reshape(features.shape[0], -1)
                            # features_t = features_t.permute(1, 0, 2)
                            # features_t = features_t.reshape(features_t.shape[0], -1)
                            # loss_feature = self.args.beta * dist_criterion(features, features_t)
                            # loss_feature = self.args.beta * angle_criterion(features, features_t)

                            # cos distance
                            # features = features.reshape(features.shape[0], -1)
                            # features_t = features_t.reshape(features_t.shape[0], -1)
                            # print((1 - torch.sum(features * features_t, dim=-1) / (torch.norm(features, dim=-1) * torch.norm(features_t, dim=-1) + 0.000001)).shape)
                            # loss_feature = self.args.beta * torch.mean((1 - torch.sum(features * features_t, dim=-1) / (torch.norm(features, dim=-1) * torch.norm(features_t, dim=-1) + 0.000001)))
                        else:
                            loss_feature = 0

                        # print(batch_x.shape) # [16, 336, 321]
                        # teacher_grad = torch.autograd.grad(criterion(outputs_t, batch_y), batch_x)[0].detach().clone()
                        # student_grad = torch.autograd.grad(criterion(outputs, batch_y), batch_x, create_graph=True)[0].detach().clone()
                        # teacher_grad = torch.autograd.grad(criterion(outputs_t, batch_y), batch_x)[0].detach().clone()

                        # teacher_grad = torch.autograd.grad(criterion(outputs_t, batch_y), model_parameters_t)
                        # teacher_grad = list((_.detach().clone() for _ in teacher_grad))
                        # student_grad = torch.autograd.grad(criterion(outputs, batch_y), model_parameters, create_graph=True)
                        # for item in teacher_grad:
                        #     print("t:", item.shape)
                        #     print(item)
                        # for item in student_grad:
                        #     print(item.shape)
                        #     print(item)
                        # print(teacher_grad)
                        # print(student_grad)

                        loss_gt = criterion(outputs, batch_y)

                        if self.args.gamma:
                            # relation kd
                            # features_r = features[-2].permute(1, 0, 2)
                            # features_r = features_r.reshape(features_r.shape[0], -1)

                            # features_t_r = features_t_reg.permute(1, 0, 2)
                            # features_t_r = features_t_r.reshape(features_t_r.shape[0], -1)
                            # loss_grad = self.args.gamma * dist_criterion(features_r, features_t_r)
                            # loss_feature = self.args.beta * angle_criterion(features, features_t)

                            # gradient matching
                            # teacher_grad = torch.autograd.grad(criterion(outputs_t, batch_y), batch_x)[0].detach().clone()
                            # student_grad = torch.autograd.grad(loss_gt, batch_x, create_graph=True)[0]

                            # # teacher_grad = torch.autograd.grad(outputs_t, batch_x, grad_outputs=torch.ones_like(outputs_t))[0].detach().clone()
                            # # student_grad = torch.autograd.grad(outputs, batch_x, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

                            # teacher_grad = teacher_grad.reshape(teacher_grad.shape[0], -1)
                            # student_grad = student_grad.reshape(student_grad.shape[0], -1)
                            # teacher_grad = F.normalize(teacher_grad, p=2, dim=-1)
                            # student_grad = F.normalize(student_grad, p=2, dim=-1)

                            # # print("t:", teacher_grad)
                            # # print("s:", student_grad)
                            # # loss_grad = criterion(teacher_grad, student_grad)

                            # loss_grad = self.args.gamma * criterion(teacher_grad, student_grad)
                            # # length = torch.sum(teacher_grad[:, :]!=0)
                            # # loss_grad = loss_grad / length

                            # # loss_grad = distance_wb(teacher_grad, student_grad)
                            # # teacher_grad = teacher_grad.reshape((-1))
                            # # student_grad = student_grad.reshape((-1))
                            # # loss_grad = 1 - torch.sum(teacher_grad * student_grad, dim=-1) / (torch.norm(teacher_grad, dim=-1) * torch.norm(student_grad, dim=-1) + 0.000001)


                            # # print(loss_grad)
                            # # length = torch.sum(teacher_grad[:,:,0]!=0)
                            # # print(length)
                            # # loss_grad = loss_grad/length
                            # # print(loss_grad)

                            # # loss = criterion(outputs, batch_y) + self.args.alpha * loss_logit + self.args.beta * loss_feature + self.args.gamma * loss_grad
                            # # print(loss_grad)
                            loss_grad = 0
                        else:
                            loss_grad = 0

                        # loss = loss_gt

                        # if self.args.alpha:
                        #     loss = loss + self.args.alpha * loss_logit
                        # if self.args.beta:
                        #     loss = loss + self.args.beta * loss_feature
                        # if self.args.gamma:
                        #     loss = loss + self.args.gamma * loss_grad
                        #     train_loss_grad.append(loss_grad.item())

                        loss = loss_gt + loss_logit + loss_feature + loss_grad
                        # loss = loss_logit + loss_feature

                    else:
                        loss = criterion(outputs, batch_y)
                        if self.args.model=="Pathformer":
                            loss = loss + balance_loss

                    if self.args.model == 'Fredformer' and self.weight_decay > 0:
                        loss = loss + 0.01 * self.reg_loss(self.model)

                    train_loss.append(loss.item())
                    if self.args.distillation:
                        train_loss_gt.append(loss_gt.item())
                        if self.args.alpha:
                            train_loss_logit.append(loss_logit.item()/self.args.alpha)
                        if self.args.beta:
                            train_loss_feature.append(loss_feature.item()/self.args.beta)
                        if self.args.gamma:
                            train_loss_grad.append(loss_grad.item()/self.args.gamma)

                if (i + 1) % 100 == 0:
                    if self.args.distillation:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss_gt: {3:.7f}".format(\
                                i + 1, epoch + 1, loss.item(), loss_gt.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.model == 'CARD' and h_loss != 0:
                        loss = loss
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler=scheduler)
                    scheduler.step()

                batch_times.append(time.time() - batch_time)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Avg batch cost time: {}".format(sum(batch_times) / len(batch_times)))
            train_loss = np.average(train_loss)
            if self.args.distillation:
                train_loss_logit = np.average(train_loss_logit)
                train_loss_feature = np.average(train_loss_feature)
                train_loss_grad = np.average(train_loss_grad)
                train_loss_gt = np.average(train_loss_gt)

            if self.args.model == 'CARD':
                vali_loss = self.vali(vali_data, vali_loader, c, is_test = False)
                test_loss = self.vali(test_data, test_loader, nn.MSELoss(), is_test = True)
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Val Loss: {3:.7f} | Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if self.args.distillation:
                print("Epoch: {0}, Steps: {1} | Train Loss GT: {2:.7f} | Train Loss Logit: {3:.7f} | Train Loss Feature: {4:.7f} | Train Loss Gamma: {5:.7f}".format(
                    epoch + 1, train_steps, train_loss_gt, train_loss_logit, train_loss_feature, train_loss_grad))
            # early_stopping(vali_loss, self.model, path, self.regressor)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.args.distillation:
            os.remove(best_model_path)
            os.rmdir(path)
            print("delete best model path:", best_model_path)

        return self.model

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.model == 'Fredformer':
                    self.select_regularization()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model=='Pathformer':
                        outputs, balance_loss, features = self.model(batch_x)
                    elif self.args.model in ['iTransformer', 'CARD']:
                        outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, features = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.model == 'CARD' and is_test == False:
                    ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                    ratio = torch.tensor(ratio).unsqueeze(-1).to('cuda')
                    outputs = outputs * ratio
                    batch_y = batch_y * ratio

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        batch_times = []
        self.model.eval()
        max_gpu_memory_usage = 0
        max_cpu_memory_usage = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    batch_time = time.time()
                    if self.args.model=='Pathformer':
                        outputs, balance_loss = self.model(batch_x)
                    elif self.args.model in ['iTransformer', 'CARD']:
                        outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, features = self.model(batch_x)
                    batch_times.append(time.time() - batch_time)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    # GPU and CPU memory usage
                    free_memory, total_memory = torch.cuda.mem_get_info(device=batch_x.device)
                    if total_memory - free_memory > max_gpu_memory_usage:
                        max_gpu_memory_usage = total_memory - free_memory
                    memory_info = psutil.virtual_memory()
                    used_memory = memory_info.used
                    if used_memory > max_cpu_memory_usage:
                        max_cpu_memory_usage = used_memory
                        
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print("Avg batch cost time: {}".format(sum(batch_times) / len(batch_times)))
        print("Max GPU memory usage: {} GB".format(max_gpu_memory_usage / (1024 ** 3)))
        print("Max CPU memory usage: {} GB".format(max_cpu_memory_usage / (1024 ** 3)))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def train_proj(self, setting=None):
        train_data, train_loader = self._get_data(flag='train')

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(self.mlp1.parameters(), lr=self.args.learning_rate_proj, weight_decay=self.args.weight_decay)
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':
            train_steps = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate_proj)
        else:
            scheduler = None

        # losses = {'loss': [], 'loss_gt': [], 'loss_feature': []}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_grad = []
            train_loss_gt = []
            train_loss_feature = []

            epoch_time = time.time()
            batch_times = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_time = time.time()
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model_t=='Pathformer':
                    outputs, balance_loss, features_t = self.model_t(batch_x)
                elif self.args.model_t in ['iTransformer', 'CARD']:
                    outputs, features_t = self.model_t(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, features_t = self.model_t(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                features_t = features_t[:, :batch_x.shape[-1], f_dim:]

                features = self.mlp1(outputs.permute(0, 2, 1))
                features_t = features_t.reshape(features_t.shape[0], features_t.shape[1], -1)
                # features = self.mlp1(outputs.permute(0, 2, 1))
                # features_t = self.mlp2(outputs_t.permute(0, 2, 1))
                loss_feature = criterion(features, features_t)
                # features = features.reshape(features.shape[0], -1)
                # features_t = features_t.reshape(features_t.shape[0], -1)
                # print((1 - torch.sum(features * features_t, dim=-1) / (torch.norm(features, dim=-1) * torch.norm(features_t, dim=-1) + 0.000001)).shape)
                # loss_feature = self.args.beta * torch.mean((1 - torch.sum(features * features_t, dim=-1) / (torch.norm(features, dim=-1) * torch.norm(features_t, dim=-1) + 0.000001)))

                # loss_gt = criterion(outputs, batch_y)
                loss_gt = criterion(outputs, batch_y)

                loss = loss_feature

                train_loss.append(loss.item())
                train_loss_gt.append(loss_gt.item())
                train_loss_feature.append(loss_feature.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss_gt: {3:.7f} | loss_feature: {4:.7f}".format(i + 1, epoch + 1, loss.item(), loss_gt.item(), loss_feature.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate_proj(model_optim, epoch + 1, self.args, scheduler=scheduler)
                    scheduler.step()

                batch_times.append(time.time() - batch_time)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Avg batch cost time: {}".format(sum(batch_times) / len(batch_times)))
            train_loss = np.average(train_loss)
            train_loss_gt = np.average(train_loss_gt)
            train_loss_feature = np.average(train_loss_feature)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Train Loss GT: {3:.7f} | Train Loss Feature: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, train_loss_gt, train_loss_feature))

            if self.args.lradj != 'TST':
                adjust_learning_rate_proj(model_optim, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            self.mlp1.eval()

        return self.mlp1
