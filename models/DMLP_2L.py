import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


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
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.configs = configs

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.ModuleList([
                                        nn.Linear(self.seq_len, self.d_model),
                                        nn.ReLU(),
                                        nn.Linear(self.d_model, self.pred_len),
                                    ])
            self.Linear_Trend = nn.ModuleList([
                                        nn.Linear(self.seq_len, self.d_model),
                                        nn.ReLU(),
                                        nn.Linear(self.d_model, self.pred_len)
                                    ])

            self.Linear_Seasonal[0].weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
            self.Linear_Seasonal[2].weight = nn.Parameter(
                (1 / self.d_model) * torch.ones([self.pred_len, self.d_model]))
            self.Linear_Trend[0].weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
            self.Linear_Trend[2].weight = nn.Parameter(
                (1 / self.d_model) * torch.ones([self.pred_len, self.d_model]))

            self.mlp = nn.ModuleList()
            self.mlp.append(nn.Linear(self.d_model, self.d_model))
            for i in range(1, 1):
                self.mlp.append(nn.Linear(self.d_model, self.d_model))

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            x_seasonal = seasonal_init
            x_seasonal1 = self.Linear_Seasonal[0](x_seasonal)
            x_seasonal2 = self.Linear_Seasonal[1](x_seasonal1)
            seasonal_output = self.Linear_Seasonal[2](x_seasonal2)

            x_trend = trend_init
            x_trend1 = self.Linear_Trend[0](x_trend)
            x_trend2 = self.Linear_Trend[1](x_trend1)
            trend_output = self.Linear_Trend[2](x_trend2)

            # feature = x_seasonal2 + x_trend2
            # for layer in self.mlp[:-1]:
            #     feature = F.relu(layer(feature))
            # feature = self.mlp[-1](feature) # torch.Size([16, 321, 512])

        # feature = seasonal_output + trend_output
        # for layer in self.mlp[:-1]:
        #     feature = F.relu(layer(feature))
        # feature = self.mlp[-1](feature) # torch.Size([16, 321, 512])

        x = seasonal_output + trend_output

        # return x.permute(0, 2, 1), feature
        return x.permute(0, 2, 1), x_seasonal2 + x_trend2

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

    def forward(self, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
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
