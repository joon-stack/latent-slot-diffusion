import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionLayerWithActivation(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn=F.relu):
        super(ProjectionLayerWithActivation, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_fn = activation_fn
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activation_fn(x)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x

class LatentPredictor(nn.Module):
    def __init__(self, slot_size, latent_size, num_layers=6, nhead=8):
        super(LatentPredictor, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_size, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.slot_proj_layer = ProjectionLayerWithActivation(slot_size, latent_size)
        self.bert_proj_layer = ProjectionLayerWithActivation(768, latent_size)
        self.backtoslot_layer = ProjectionLayer(latent_size, slot_size)

    def forward(self, memory, tgt):
        memory = self.bert_proj_layer(memory)
        tgt = self.slot_proj_layer(tgt)
        out = self.transformer_decoder(tgt, memory)
        out = self.backtoslot_layer(out)
        
        return out
