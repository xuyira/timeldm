import torch
import random
import argparse
import numpy as np

from Engine.solver import Trainer
from Data.build_dataloader import build_dataloader
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')

    # args for random

    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    
    # args for dataset
    
    parser.add_argument('--dataset', type=str, default='sine',
                        choices=['sine', 'energy', 'etth', 'mujoco', 'fmri', 'stock'],
                        help='Name of Selected Time Series Dataset.')
    parser.add_argument('--seq_length', type=int, default=24,
                        help='Length of Time Series.')
    parser.add_argument('--feat_dim', type=int, default=5,
                        help='Number of Variables in Time Series.')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Size of Synthetic Dataset.')
    parser.add_argument('--proportion', type=float, default=1.,
                        help='Proportion of Training Data in Real-world Dataset.')
    parser.add_argument('--data_root', type=str, default='./Data/datasets/.csv')
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
    
    # args for diffusion

    parser.add_argument('--time_steps', type=int, default=500,
                        help='Number of Diffusion Steps.')
    parser.add_argument('--sample_steps', type=int, default=500,
                        help='Number of Sampling Steps.')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2'], help='Type of Loss Function.')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Type of Beta Schedule.')
    
    # args for models

    parser.add_argument('--n_layer_enc', type=int, default=1,
                        help='Number of Encoder Layers.')
    parser.add_argument('--n_layer_dec', type=int, default=2,
                        help='Number of Decoder Layers.')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of Attention Heads')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Dimmension of Transformer.')
    parser.add_argument('--mlp_times', type=int, default=4,
                        help='Scale Times of MLP Hidden Dimmension in Transformer.')
    
    # args for training

    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='Learning Rate before Warmup.')
    parser.add_argument('--warmup_lr', type=float, default=8e-4,
                        help='Learning Rate after Warmup.')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum Learning Rate.')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Number of Warmup Epochs.')
    parser.add_argument('--patience', type=int, default=3000,
                        help='Patience.')
    parser.add_argument('--threshold', type=float, default=1e-1,
                        help='Hyperparameter for Evaluating whether Better or not.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Hyperparameter for Reducing Learning Rate.')
    parser.add_argument('--ema_cycle', type=int, default=10,
                        help='Number of Epochs between Two EMA Updating.')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='Decay Rate of EMA.')
    parser.add_argument('--train_epochs', type=int, default=12000,
                        help='Number of Training Epochs.')
    parser.add_argument('--accumulate_cycle', type=int, default=2,
                        help='Number of Epochs between Two Gradient Descent.')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='Train or Test.')

    args = parser.parse_args()

    return args

def main(milestone=None):
    args = parse_args()
    setup_seed(args.seed)
    dataloader, dataset = build_dataloader(name=args.dataset, batch_size=args.batch_size, window=args.seq_length, dim=args.feat_dim,
                                           num_samples=args.num_samples, data_root=args.data_root, proportion=args.proportion)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    diffusion = Diffusion_TS(feature_size=args.feat_dim, n_layer_enc=args.n_layer_enc, n_layer_dec=args.n_layer_dec, seq_length=args.seq_length,
                             d_model=args.d_model, sampling_timesteps=args.sample_steps, timesteps=args.time_steps, loss_type=args.loss_type,
                             beta_schedule=args.beta_schedule, n_heads=args.n_heads, mlp_hidden_times=args.mlp_times).to(device)
    
    trainer = Trainer(model=diffusion, data_loader=dataloader, results_folder=f'./Checkpoints_{args.dataset}', train_lr=args.base_lr,
                      train_num_steps=args.train_epochs, gradient_accumulate_every=args.accumulate_cycle, min_lr=args.min_lr,
                      ema_update_every=args.ema_cycle, ema_decay=args.ema_decay, patience=args.patience,
                      threshold=args.threshold, warmup=args.warmup, factor=args.factor, warmup_lr=args.warmup_lr)
    
    if args.is_train:
        trainer.train()
    else:
        trainer.load(milestone)

    samples = trainer.sample(num=len(dataset), size_every=2001, shape=[args.seq_length, args.feat_dim])
    if dataset.auto_norm:
        samples = unnormalize_to_zero_to_one(samples)
        np.save(f'./OUTPUT/samples/ddpm_fake_{args.dataset}.npy', samples)


if __name__ == '__main__':
    main()