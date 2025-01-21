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
import copy
from models import iTransformer, MLP, Regressor
from torch.optim import lr_scheduler
import json
import glob
import psutil
from layers.Autoformer_EncDec import series_decomp

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


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
            model_t_names = args.model_t.split(' ')
            self.model_t = []
            for model_t_name in model_t_names:
                print(model_t_name)
                args_t = copy.deepcopy(args)
                with open('config.json', 'r') as f:
                    config = json.load(f)
                model_config = config[self.args.model_id.split('_')[0]][model_t_name]
                for key, value in model_config.items():
                    setattr(args_t, key, value)
                self.args_t = args_t
                model_t = self.model_dict[model_t_name].Model(args_t).float().to(self.device)
                print("path:", f"./checkpoints/long_term_forecast_{self.args.model_id}_{model_t_name}_{args.data}")
                checkpoint_path = glob.glob(f"./checkpoints/long_term_forecast_{self.args.model_id}_{model_t_name}_{args.data}*/checkpoint.pth")[0]
                print("teacher checkpoint_path:", checkpoint_path)
                model_t.load_state_dict(torch.load(checkpoint_path))
                model_t.eval()
                self.model_t.append(model_t)
            if self.args.model_t in ['PatchTST']:
                self.regressor = Regressor.Model(args, args_t.d_model * int((args.seq_len - 16) / 8 + 2), args.d_model).float().to(self.device)
            elif self.args.model_t in ['FEDformer', 'Autoformer']:
                self.regressor = Regressor.Model(args, args.seq_len, args.d_model).float().to(self.device)
            elif self.args.model_t in ['ModernTCN']:
                self.regressor = Regressor.Model(args, args_t.dims[-1] * (args.seq_len // args_t.patch_stride), args.d_model).float().to(self.device)
            elif self.args.model_t in ['TimeMixer']:
                self.regressor = Regressor.Model(args, args.pred_len, args.d_model).float().to(self.device)
            else:
                self.regressor = Regressor.Model(args, args_t.d_model, args.d_model).float().to(self.device)
            
            if self.args.model in ['TSMixer']:
                self.regressor = Regressor.Model(args, args_t.dims[-1] * (args.seq_len // args_t.patch_stride), args.seq_len).float().to(self.device)

            self.regressor.train()


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        count_parameters(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.distillation:
            model_optim = optim.Adam(list(self.model.parameters()) + list(self.regressor.parameters()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def train(self, setting):
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

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None

                    outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.distillation:
                        with torch.no_grad():
                            outputs_t, features_t = [], []
                            for model_t in self.model_t:
                                outputs_t_tmp, features_t_tmp = model_t(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                f_dim = -1 if self.args.features == 'MS' else 0
                                outputs_t_tmp = outputs_t_tmp[:, -self.args.pred_len:, f_dim:]
                                if self.args.model_t in ['PatchTST']:
                                    features_t_tmp = torch.reshape(features_t_tmp[-2], (-1,batch_x.shape[-1], features_t_tmp[-2].shape[-2], features_t_tmp[-2].shape[-1]))
                                    features_t_tmp = nn.Flatten(start_dim=-2)(features_t_tmp)
                                elif self.args.model_t in ['FEDformer', 'Autoformer']:
                                    features_t_tmp = features_t_tmp[-2].permute(0, 2, 1)
                                    features_t_tmp = features_t_tmp[:, :batch_x.shape[-1], f_dim:]
                                    if features[-2].shape[1] > features_t_tmp.shape[1]:
                                        features[-2] = features[-2][:, :features_t_tmp.shape[1], :]
                                elif self.args.model_t in ['Crossformer']:
                                    features_t_tmp = features_t_tmp[-2][0].mean(dim=-2)
                                elif self.args.model_t in ['TimesNet']:
                                    features_t_tmp[-1] = features_t_tmp[-1].permute(0, 2, 1)
                                    features_t_tmp = features_t_tmp[-1][:, :, -self.args_t.d_model:]
                                elif self.args.model_t in ['MICN']:
                                    features_t_tmp = features_t_tmp[-2].permute(0, 2, 1)
                                    features_t_tmp = features_t_tmp[:, :, -self.args_t.d_model:]
                                elif self.args.model_t in ['ModernTCN']:
                                    features_t_tmp = nn.Flatten(start_dim=-2)(features_t_tmp[-2])
                                elif self.args.model_t in ['TimeMixer']:
                                    features_t_tmp = features_t_tmp[-1].permute(0, 2, 1)
                                else:
                                    features_t_tmp = features_t_tmp[-2][:, :batch_x.shape[-1], f_dim:]
                                outputs_t.append(outputs_t_tmp)
                                features_t.append(features_t_tmp)

                        # logit level
                        outputs_t = outputs_t[0]

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

                        # feature level 
                        if len(features_t) == 1:
                            features_t = features_t[0]
                            features_t = features_t.reshape(features_t.shape[0], features_t.shape[1], -1) # [32, 8, 325, 325]
                            features_t_reg = self.regressor(features_t)

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

                        loss_feature = loss_feature_frequency + loss_feature_multiscale

                        loss_gt = criterion(outputs, batch_y)

                        loss = loss_gt + loss_logit + loss_feature
                    else:
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())
                    if self.args.distillation:
                        train_loss_gt.append(loss_gt.item())
                        if self.args.alpha:
                            train_loss_logit.append(loss_logit.item()/self.args.alpha)
                        if self.args.beta:
                            train_loss_feature.append(loss_feature.item()/self.args.beta)

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
                if self.args.alpha:
                    train_loss_logit = np.average(train_loss_logit)
                if self.args.beta:
                    train_loss_feature = np.average(train_loss_feature)
                train_loss_gt = np.average(train_loss_gt)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Val Loss: {3:.7f} | Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if self.args.distillation:
                if self.args.alpha and self.args.beta:
                    print("Epoch: {0}, Steps: {1} | Train Loss GT: {2:.7f} | Train Loss Logit: {3:.7f} | Train Loss Feature: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss_gt, train_loss_logit, train_loss_feature))
                else:
                    print("Epoch: {0}, Steps: {1} | Train Loss GT: {2:.7f}".format(
                        epoch + 1, train_steps, train_loss_gt))

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
        # if self.args.distillation:
        #     os.remove(best_model_path)
        #     os.rmdir(path)
        #     print("delete best model path:", best_model_path)

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

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None

                    outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

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
            print("path:", f"./checkpoints/long_term_forecast_{self.args.model_id}_{self.args.model}_{self.args.data}")
            if self.args.distillation:
                checkpoint_path = glob.glob(f"./checkpoints/long_term_forecast_{self.args.model_id}_{self.args.model}_{self.args.model_t}_{self.args.data}*/checkpoint.pth")[0]
            else:
                checkpoint_path = glob.glob(f"./checkpoints/long_term_forecast_{self.args.model_id}_{self.args.model}_{self.args.data}*/checkpoint.pth")[0]
            print("checkpoint_path:", checkpoint_path)
            print(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(checkpoint_path))

        inputs = []
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
                    if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                        
                    batch_time = time.time()
                    
                    outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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

                inputs.append(batch_x.detach().cpu().numpy())
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
        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        return