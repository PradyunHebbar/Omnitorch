import torch
import torch.nn as nn

from .PET import PETBody, ClassifierHead, GeneratorHead

class PET(nn.Module):
    def __init__(self,
                 num_feat,
                 num_jet,
                 num_classes=10,
                 num_keep=7,
                 feature_drop=0.1,
                 projection_dim=128,
                 local=True,
                 K=10,
                 num_local=2,
                 num_layers=8,
                 num_class_layers=2,
                 num_gen_layers=2,
                 num_heads=4,
                 drop_probability=0.0,
                 simple=False,
                 layer_scale=True,
                 layer_scale_init=1e-5,
                 talking_head=False,
                 mode='all',
                 num_diffusion=3,
                 dropout=0.0,
                 device='cuda'
                # no class activation option
                ):
        super().__init__()
        self.device = device
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.drop_probability = drop_probability
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        self.mode = mode
        self.num_diffusion = num_diffusion
        self.num_local = num_local
        self.ema = 0.999

        self.body = PETBody(num_feat, num_keep, feature_drop, projection_dim, local, K, num_local,
                            num_layers, num_heads, drop_probability, talking_head, layer_scale,
                            layer_scale_init, dropout, device, mode
                           #self.LocalEmbedding , self.TransformerBlock, self.FourierProjection
                           )

        self.classifier_head = ClassifierHead(
            projection_dim, num_jet, num_classes, num_class_layers, simple,
            num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability
        #self.ClassifierTransformerBlock
        )

        self.generator_head = GeneratorHead(
            projection_dim, num_jet, num_classes, num_feat, num_gen_layers, simple,
            num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability,
            feature_drop
        #self.GeneratorTransformerBlock, self.FourierProjection
        )

        self.ema_body = nn.Module()
        self.ema_generator_head = nn.Module()

    def forward(self, x, mode='all'):
        body_output = self.body(x)

        if mode in ['classifier', 'all']:
            classifier_output = self.classifier_head(body_output, x['input_jet'], x['input_mask'])

        if mode in ['generator', 'all']:
            generator_output = self.generator_head(body_output, x['input_jet'], x['input_mask'],
                                                   x['input_time'], x['input_label'])
        if mode == 'classifier':
            return classifier_output
        elif mode == 'generator':
            return generator_output
        else:
            return classifier_output, generator_output
