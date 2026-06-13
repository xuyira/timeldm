import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP
import math
from einops import rearrange, reduce, repeat

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization = 'kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression = None, value_compression = None):
  
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a/b , dim=-1)

        
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x
        
class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout = 0.0,
        ffn_dropout = 0.0,
        residual_dropout = 0.0,
        activation = 'relu',
        prenormalization = True,
        initialization = 'kaiming',      
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)


    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )
            x = self._end_residual(x, x_residual, layer, 0)
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x
    
class Transformer_dec(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout = 0.0,
        ffn_dropout = 0.0,
        residual_dropout = 0.0,
        activation = 'relu',
        prenormalization = True,
        initialization = 'kaiming',      
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention1': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'attention2': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)


    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x,rec):
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention1'](
              
                x_residual,
                x_residual,
            )
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention2'](
               
                x_residual,
                rec,
            )
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
       
class AE(nn.Module):
    def __init__(self, hid_dim, n_head):
        super(AE, self).__init__()
 
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.encoder = MultiheadAttention(hid_dim, n_head)
        self.decoder = MultiheadAttention(hid_dim, n_head)
    def get_embedding(self, x):
        return self.encoder(x, x).detach() 
    def forward(self, x):
        z = self.encoder(x, x)
        h = self.decoder(z, z)
        return h

class VAE(nn.Module):
    def __init__(self,num_layers, hid_dim, n_head = 1, factor = 4,length = 1024, indim =5):
        super(VAE, self).__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.pos_mu = LearnablePositionalEncoding(hid_dim, dropout=0.0, max_len= length)
        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.pos_log = LearnablePositionalEncoding(hid_dim, dropout=0.0, max_len= length)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.pos_dec = LearnablePositionalEncoding(hid_dim, dropout=0.0, max_len= length)
        self.decoder = Transformer_dec(num_layers, hid_dim, n_head, hid_dim, factor)
        self.dec = Conv_MLP(hid_dim, hid_dim, resid_pdrop=0)

    def get_embedding(self, x):
        return self.encoder_mu(x, x).detach() 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x ):
        mu_x = self.pos_mu(x)
        mu_z = self.encoder_mu(mu_x)
        std_z = self.encoder_logvar(mu_x)
        z = self.reparameterize(mu_z, std_z)
        zx = self.dec(z)
        ind = self.pos_dec(zx)
        h = self.decoder(ind,z)
        return mu_z, std_z, h
    
class Model_VAE(nn.Module):
    def __init__(self, num_layers, d_token, n_head = 1, factor = 4, in_dim=5, length =1024):
        super(Model_VAE, self).__init__()
        self.VAE = VAE(num_layers, d_token, n_head = n_head, factor = factor, length= length, indim = in_dim)
        self.emb = Conv_MLP(in_dim, d_token, resid_pdrop=0)
        self.rec = nn.Linear(d_token, in_dim)
                            
    def forward(self, x_num):
        # print(x_num.shape)
        emb = self.emb(x_num)
        mu_z, std_z, h  = self.VAE(emb)
        out = self.rec(h)
        return out, mu_z, std_z

class Encoder_model(nn.Module):
    def __init__(self, num_layers,  d_token, n_head, factor, in_dim, length):
        super(Encoder_model, self).__init__()
        self.emb = Conv_MLP(in_dim, d_token, resid_pdrop=0)
        self.VAE_Encoder_mu = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.VAE_Encoder_logvar = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.pos_mu = LearnablePositionalEncoding(d_token, dropout=0.0, max_len= length)
        self.pos_log = LearnablePositionalEncoding(d_token, dropout=0.0, max_len= length)

    def load_weights(self, Pretrained_VAE):
        self.emb.load_state_dict(Pretrained_VAE.emb.state_dict())
        self.VAE_Encoder_mu.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())
        self.VAE_Encoder_logvar.load_state_dict(Pretrained_VAE.VAE.encoder_logvar.state_dict())
        self.pos_mu.load_state_dict(Pretrained_VAE.VAE.pos_mu.state_dict())
        self.pos_log.load_state_dict(Pretrained_VAE.VAE.pos_log.state_dict())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):

        x = self.emb(x)
        mu_x = self.pos_mu(x)
        mu = self.VAE_Encoder_mu(mu_x)
        var = self.VAE_Encoder_logvar(mu_x)
        z = self.reparameterize(mu, var)
        return z

class Decoder_model(nn.Module):
    def __init__(self, num_layers,  d_token, n_head, factor, in_dim, length):
        super(Decoder_model, self).__init__()
        self.pos_dec = LearnablePositionalEncoding(d_token, dropout=0.0, max_len= length)
        
        self.VAE_Decoder = Transformer_dec(num_layers, d_token, n_head, d_token, factor)
        self.dec = Conv_MLP(d_token, d_token, resid_pdrop=0)
        self.rec = nn.Linear(d_token, in_dim)
        
    def load_weights(self, Pretrained_VAE):
        self.pos_dec.load_state_dict(Pretrained_VAE.VAE.pos_dec.state_dict())
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.rec.load_state_dict(Pretrained_VAE.rec.state_dict())
        self.dec.load_state_dict(Pretrained_VAE.VAE.dec.state_dict())
       
    def forward(self, z):
        dec = self.dec(z)
        ind = self.pos_dec(dec)
        h = self.VAE_Decoder(ind,z)
        out = self.rec(h)

        return out


