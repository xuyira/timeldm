import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import pathlib
import numpy as np
from tqdm import tqdm
from LDMs.model import MLPDiffusion, Model
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Pipeline')
parser.add_argument('--seq_length', type=int, default=24,
                    help='Length of Time Series.')
parser.add_argument('--feat_dim', type=int, default=7,
                    help='Number of Variables in Time Series.')                  
parser.add_argument('--LDM_dim', type=int, default=1024, help='N.')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10000, help='batch size')
parser.add_argument('--dataset', type=str, default='sine',choices=['sine', 'energy', 'etth', 'mujoco', 'fmri', 'stock'], help='Name of Selected Time Series Dataset.')
parser.add_argument('--embedding_save_path', type=str, default='', help='batch size')
parser.add_argument('--D_TOKEN', type=int, default=32,help='D_TOKEN')
parser.add_argument('--N_HEAD', type=int, default=1,help='N_HEAD')
parser.add_argument('--FACTOR', type=int, default=16,help='FACTOR')
parser.add_argument('--NUM_LAYERS', type=int, default=2,help='NUM_LAYERS')                  
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')
curr_dir = os.path.dirname(os.path.abspath(__file__))

ckpt_path = f'{curr_dir}/ckpt_ldm/ldm_{args.dataset}_{args.D_TOKEN}_{args.N_HEAD}_{args.FACTOR}_{args.NUM_LAYERS}' 
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

data_z = os.path.dirname(os.path.abspath(__file__))
data_z = f'{data_z}/ckpt_vae/{args.dataset}_{args.D_TOKEN}_{args.N_HEAD}_{args.FACTOR}_{args.NUM_LAYERS}/train_z.npy' 
warnings.filterwarnings('ignore')

LOG = os.path.join(pathlib.Path(__file__).parent.resolve(), 'Logs')
LOG_DIR = os.path.join(LOG, f'ldm_{args.dataset}_{args.D_TOKEN}_{args.N_HEAD}_{args.FACTOR}_{args.NUM_LAYERS}')
os.makedirs(LOG_DIR, exist_ok=True)

writer = SummaryWriter(log_dir=LOG_DIR)

def main(): 
    
    embedding_save_path = data_z
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    bat, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(bat, in_dim)
    train_data = train_z
    print('train_data', train_data.shape)
    train_loader = DataLoader(
        train_data,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 0,
    )

    denoise_fn = MLPDiffusion(in_dim, args.LDM_dim).to(device)
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, betas=(0.9,0.96))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)
   
    best_loss = float('inf')
    patience = 0
   
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")

        batch_loss = 0.0
        len_input = 0

        for batch in pbar:
            model.train()
            inputs = batch.float().to(device)
            loss = model(inputs)
            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": loss.item()})
        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)
        writer.add_scalar('loss/curr_loss', curr_loss, epoch)
        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 8000:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

if __name__ == '__main__':
    main()



