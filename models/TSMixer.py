import torch.nn as nn
from layers.StandardNorm import Normalize

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.revin_layer = Normalize(num_features=configs.enc_in, affine=True, subtract_last=False)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = self.revin_layer(x_enc, 'norm')

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        feature = []
        feature.append(x_enc.transpose(1, 2))
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        feature.append(enc_out)

        enc_out = self.revin_layer(enc_out, 'denorm')
        
        return enc_out, feature

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, feature = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], feature  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
