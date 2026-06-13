import torch
import warnings
import numpy as np
from LDMs.model import MLPDiffusion, Model
from LDMs.vae.vae_model import Decoder_model
from LDMs.diffusion_utils import sample
import argparse
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='PyTorch Training script')
parser.add_argument('--feat_dim', type=int, default=50,help='Number of Variables in Time Series.')                  
parser.add_argument('--D_TOKEN', type=float, default=100,help='D_TOKEN')
parser.add_argument('--N_HEAD', type=float, default=1,help='N_HEAD')
parser.add_argument('--FACTOR', type=float, default=16,help='FACTOR')
parser.add_argument('--NUM_LAYERS', type=float, default=2,help='NUM_LAYERS')
parser.add_argument('--seq_length', type=int, default=24,help='seq_length') 
parser.add_argument('--LDM_dim', type=int, default=4096,help='LDM_dim')
parser.add_argument('--embedding_save_path', type=str, default='./ckpt_vae/fmri_100_1_16_2/train_z.npy', help='embedding_save_path')
parser.add_argument('--MLPDiffusion', type=str, default='./ckpt_ldm/ldm_fmri_100_1_16_2/model_10000.pt')
parser.add_argument('--decoder', type=str, default='./ckpt_vae/fmri_100_1_16_2/decoder.pt')
parser.add_argument('--save_npy', type=str, default='./Data/ldm_fake_fmri_100_16_12_2.npy')             
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5
def main():
   
    embedding_save_path = args.embedding_save_path
  
    train_z = torch.tensor(np.load(embedding_save_path)).float()
    bat, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(bat, in_dim)

    denoise_fn = MLPDiffusion(in_dim, args.LDM_dim).to(device)
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(args.MLPDiffusion))

    num_samples = train_z.shape[0]
    sample_dim = in_dim
    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)

    syn_data = x_next.float().cpu().numpy()
    pre_decoder = Decoder_model(args.NUM_LAYERS,  args.D_TOKEN, n_head = args.N_HEAD, factor = args.FACTOR, in_dim = args.feat_dim,length = args.seq_length)
    decoder_save_path = args.decoder
    
    pre_decoder.load_state_dict(torch.load(decoder_save_path))
    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    out_data = pre_decoder(torch.tensor(syn_data)).detach().cpu().numpy()
    np.save(args.save_npy, out_data)
    print('out_data',out_data.shape)
    print('finish')
if __name__ == '__main__':
    main()