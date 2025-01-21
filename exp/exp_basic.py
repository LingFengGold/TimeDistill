import os
import torch
# from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
#     Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
#     Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, \
#     DMLP_2L, DMLP_3L, DMLP_4L,Linear, MLP_2L, MLP_3L, PathFormer

from models import DMLP_cd, Transformer, FEDformer, Pyraformer, Autoformer, Informer, DLinear, iTransformer, DMLP_2L, DMLP_3L, DMLP_4L, Linear, MLP, MLP_3L, Pathformer, CARD, Fredformer,\
                PatchTST, TimeMixer, SCINet, ModernTCN, DMLP,  Crossformer, TimesNet, MICN, LightTS, TSMixer, TiDE, FreTS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            'iTransformer': iTransformer,
            # 'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            # 'SegRNN': SegRNN,
            # 'TemporalFusionTransformer': TemporalFusionTransformer,
            'DMLP_2L': DMLP_2L,
            'DMLP_3L': DMLP_3L,
            'DMLP_4L': DMLP_4L,
            'DMLP': DMLP,
            'Linear': Linear,
            'MLP': MLP,
            'MLP_3L': MLP_3L,
            'Pathformer': Pathformer,
            'CARD': CARD,
            'Fredformer': Fredformer,
            "SCINet": SCINet,
            "ModernTCN": ModernTCN,
            'DMLP_cd':  DMLP_cd
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict[Mamba] = Mamba

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
