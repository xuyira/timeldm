import random
import math
import argparse
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from LDMs.vae.vae_model import Model_VAE, Encoder_model, Decoder_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import reduce
from Data.build_dataloader import build_data
from torch.utils.tensorboard import SummaryWriter

def get_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training script')

    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    # args for dataset
    parser.add_argument('--dataset', type=str, default='sine',
                        choices=['sine', 'energy', 'etth', 'mujoco', 'fmri', 'stock'],
                        help='Name of Selected Time Series Dataset.')
    parser.add_argument('--seq_length', type=int, default=24,
                        help='Length of Time Series.')
    parser.add_argument('--feat_dim', type=int, default=7,
                        help='Number of Variables in Time Series.')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Size of Synthetic Dataset.')
    parser.add_argument('--proportion', type=float, default=1.,
                        help='Proportion of Training Data in Real-world Dataset.')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--data_seed', type=int, default=123,
                        help='Seed for Data Sampling.')

    # args for irregular sampling
    parser.add_argument('--irregular', type=bool, default=False,
                        help='Irregular Time Series or not.')
    parser.add_argument('--mask_mode', type=str, default='separate',
                        help='Each Variable is Independent (separate).')
    parser.add_argument('--distribution', type=str, default='geometric',
                        choices=['geometric', 'bernoulli'],
                        help='Distribution of Masking Selection.')
    parser.add_argument('--mask_length', type=int, default=6,
                        help='Average Length of Masking Subsequences.')
    parser.add_argument('--masking_ratio', type=float, default=0.25,
                        help='Proportion of Sequence Length to be Masked.')
    # args for training
    parser.add_argument('--train_epochs', type=int, default=10000, 
                        help='Number of Training Epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training.')
    parser.add_argument('--LR', type=float, default=1e-3,help='LR')                   
    parser.add_argument('--WD', type=float, default=0,help='WD')  
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')
    parser.add_argument('--use_ff', type=bool, default=True ,help='use_ff')              
    parser.add_argument('--D_TOKEN', type=int, default=32,help='D_TOKEN')
    parser.add_argument('--N_HEAD', type=int, default=1,help='N_HEAD')
    parser.add_argument('--FACTOR', type=int, default=16,help='FACTOR')
    parser.add_argument('--NUM_LAYERS', type=int, default=2,help='NUM_LAYERS')

    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(args):
    if args.dataset == 'sine':
        build_data(name=args.dataset, 
                    window=args.seq_length, 
                    dim=args.feat_dim,
                    num_samples=args.num_samples, 
                    data_root=args.data_root, 
                    proportion=args.proportion)
        data_path = f"./Data/samples_{args.seq_length}/sine_ground_truth.npy"
    else:
        build_data(name=args.dataset,
                    window=args.seq_length, 
                    num_samples=args.num_samples, 
                    data_root=args.data_root, 
                    proportion=args.proportion)
        data_path = f"./Data/samples_{ args.seq_length}/{args.dataset}_norm_truth.npy"
    return data_path

def compute_loss(input, output, mu_z, logvar_z):
    def loss_axis(input,output, axis):
        x = torch.mean(input, dim=axis)
        y = torch.mean(output, dim=axis)
        err = (x - y).pow(2).mean()
        return err
    mse_loss = (input - output).pow(2).mean()
    l1_loss = F.l1_loss(input,output)
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    mse_loss+=loss_axis(input,output, 2)
    loss = l1_loss + mse_loss
    return loss,  loss_kld


def fft_loss(input, out):
    fft1, fft2 = torch.fft.fft(out.transpose(1, 2), norm='forward'), torch.fft.fft(input.transpose(1, 2), norm='forward')
    fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
    fourier_loss = F.l1_loss(torch.real(fft1), torch.real(fft2), reduction='none')\
                + F.l1_loss(torch.imag(fft1), torch.imag(fft2), reduction='none')
    fourier_loss = reduce(fourier_loss, 'b ... -> b (...)', 'mean').mean()
    return fourier_loss

def main(args):
    setup_seed(args.seed)
    LOG = os.path.join(pathlib.Path(__file__).parent.resolve(), 'Logs')
    LOG_DIR = os.path.join(LOG, f'vae_{args.dataset}_{args.D_TOKEN}_{args.N_HEAD}_{args.FACTOR}_{args.NUM_LAYERS}')
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt_vae/{args.dataset}_{args.D_TOKEN}_{args.N_HEAD}_{args.FACTOR}_{args.NUM_LAYERS}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = get_data(args)
    full_train_data = np.load(data_path)
   
    length = int(full_train_data.shape[0] * (8/10))
    train_data = full_train_data[:length]
    valid_data = full_train_data[length:] 
    in_train_data = torch.tensor(train_data).float().to(device)
    in_valid_data = torch.tensor(valid_data).float().to(device)

    print('in_train_data', in_train_data.shape)
    print('in_valid_data', in_valid_data.shape)

    train_loader = DataLoader(
        in_train_data,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 0,
    )

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    model = Model_VAE(args.NUM_LAYERS,  args.D_TOKEN, n_head =args. N_HEAD, factor = args.FACTOR, in_dim = args.feat_dim, length = args.seq_length).to(device)
    pre_encoder = Encoder_model(args.NUM_LAYERS, args.D_TOKEN, n_head = args.N_HEAD, factor = args.FACTOR, in_dim = args.feat_dim,length = args.seq_length).to(device)
    pre_decoder = Decoder_model(args.NUM_LAYERS,  args.D_TOKEN, n_head = args.N_HEAD, factor = args.FACTOR, in_dim = args.feat_dim, length = args.seq_length).to(device)
    
    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)
    
    best_train_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']
    patience = 0
    beta = max_beta
    for epoch in range(args.train_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{args.train_epochs}")

        total_recons_loss = 0
        total_kl_loss = 0
        total_loss = 0
        curr_count = 0

        for batch_data in pbar:
            model.train()
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            out, mu_z, std_z = model(batch_data)
            mse_loss, loss_kld = compute_loss(batch_data, out, mu_z, std_z)
            loss = mse_loss + beta * loss_kld
            fourier_loss = torch.tensor([0.])
            if args.use_ff:
                fourier_loss = fft_loss(batch_data, out)
                # fourier_loss= 0
            loss += (math.sqrt(args.seq_length) / 5) * fourier_loss  

            loss.backward()
            optimizer.step()

            writer.add_scalar('train/fourier_loss', fourier_loss, epoch)
            writer.add_scalar('train/mse_loss', mse_loss, epoch)
            writer.add_scalar('train/loss_kld', loss_kld, epoch)
            writer.add_scalar('train/loss', loss, epoch)

            batch_length = batch_data.shape[0]
            curr_count += batch_length

            total_loss += loss.item() * batch_length
            total_recons_loss += mse_loss.item() * batch_length
            total_kl_loss  += loss_kld.item() * batch_length

        all_total_loss = total_loss / curr_count
        all_total_recons_loss = total_recons_loss / curr_count
        all_total_kl_loss = total_kl_loss / curr_count
        
        writer.add_scalar('all/all_total_recons_loss', all_total_recons_loss, epoch)
        writer.add_scalar('all/all_total_kl_loss', all_total_kl_loss, epoch)
        writer.add_scalar('all/all_total_loss', all_total_loss, epoch)

        model.eval()
        with torch.no_grad():
            out, mu_z, std_z = model(in_valid_data)
            mse_loss,  loss_kld = compute_loss(in_valid_data, out, mu_z, std_z)
            val_loss = mse_loss + beta * loss_kld
            
            fourier_loss = torch.tensor([0.])
            if args.use_ff:
                fourier_loss = fft_loss(in_valid_data, out)
            val_loss += (math.sqrt(batch_data.shape[1]) / 5) * fourier_loss  
            
            writer.add_scalar('val/val_loss', val_loss, epoch)
            writer.add_scalar('val/mse_loss', mse_loss, epoch)
            writer.add_scalar('val/loss_kld', loss_kld, epoch)
            
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd
            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/model_{epoch}.pt')

        print('epoch: {}, beta = {:.6f}, Train Total: {:.6f}, Train REC:{:.6f}, Train KL:{:.6f}, Val Total:{:.6f}, Val MSE:{:.6f}, Val KL:{:6f}'.format(epoch, beta, all_total_loss, all_total_recons_loss, all_total_kl_loss, val_loss.item(), mse_loss.item(), loss_kld.item()))


    with torch.no_grad():

        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        data = torch.tensor(full_train_data).float().to(device)
        train_z = pre_encoder(data).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)
        print('train_z', train_z.shape)
        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__': 
    args = get_arg()
    main(args)
   