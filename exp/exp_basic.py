import os
import torch

from models import FEDformer, Autoformer, DLinear, iTransformer, Linear, MLP, \
                PatchTST, TimeMixer, ModernTCN, DMLP, TimesNet, MICN, LightTS, TSMixer, FreTS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'LightTS': LightTS,
            'PatchTST': PatchTST,
            'MICN': MICN,
            'iTransformer': iTransformer,
            'FreTS': FreTS,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'DMLP': DMLP,
            'Linear': Linear,
            'MLP': MLP,
            "ModernTCN": ModernTCN,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
