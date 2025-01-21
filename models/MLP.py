import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.StandardNorm import Normalize

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        # self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.configs = configs

        self.Linear_Seasonal = nn.ModuleList([
                                    nn.Linear(self.seq_len, self.d_model),
                                    nn.ReLU(),
                                    nn.Linear(self.d_model, self.pred_len),
                                ])
        # self.Linear_Trend = nn.ModuleList([
        #                             nn.Linear(self.seq_len, self.d_model),
        #                             nn.ReLU(),
        #                             nn.Linear(self.d_model, self.pred_len)
        #                         ])

        # self.Linear_Seasonal[0].weight = nn.Parameter(
        #     (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
        # self.Linear_Seasonal[2].weight = nn.Parameter(
        #     (1 / self.d_model) * torch.ones([self.pred_len, self.d_model]))
        # self.Linear_Trend[0].weight = nn.Parameter(
        #     (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
        # self.Linear_Trend[2].weight = nn.Parameter(
        #     (1 / self.d_model) * torch.ones([self.pred_len, self.d_model]))

        self.revin_layer = Normalize(num_features=configs.enc_in, affine=True, subtract_last=False)

        self.bn1 = nn.BatchNorm1d(self.configs.enc_in).to('cuda')
        # self.bn2 = nn.BatchNorm1d(self.configs.enc_in).to('cuda')

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        if self.configs.norm == 'non-stationary':
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            # means = x.mean(1, keepdim=True).detach()
            # x = x - means  # Out-of-place operation
            # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x = x / stdev  # Out-of-place operation
        elif self.configs.norm == 'revin':
            x = self.revin_layer(x, 'norm')
        features = []
        
        seasonal_init = x.permute(0, 2, 1)

        x_seasonal = seasonal_init
        x_seasonal = self.Linear_Seasonal[0](x_seasonal)

        features.append(x_seasonal)

        x_seasonal = self.bn1(x_seasonal)

        x_seasonal = self.Linear_Seasonal[1](x_seasonal)
        seasonal_output = self.Linear_Seasonal[2](x_seasonal)

        x = seasonal_output
        
        features.append(x)

        x = x.permute(0, 2, 1)
        if self.configs.norm == 'non-stationary':
            # De-Normalization from Non-stationary Transformer
            x = x + means
            # x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # Out-of-place operation
            # x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # Out-of-place operation
        elif self.configs.norm == 'revin':
            x = self.revin_layer(x, 'denorm')
        
        return x, features

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, features = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :] , features # dec_out: [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
