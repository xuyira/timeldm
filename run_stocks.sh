########################################################
CUDA_VISIBLE_DEVICES=0 \
python train_vae.py \
--data_root './Data/datasets/stock_data.csv' \
--dataset stock \
--train_epochs 15001 \
--seq_length 24 \
--feat_dim 6 \
--D_TOKEN 12 \
--N_HEAD 2 \
--FACTOR 16 \
--NUM_LAYERS 3 \
--batch_size 512

python train_ldm.py \
--dataset stock \
--num_epochs 15000 \
--seq_length 24 \
--feat_dim 6 \
--D_TOKEN 12 \
--N_HEAD 2 \
--FACTOR 16 \
--NUM_LAYERS 3 \
--LDM_dim 1024 \
--batch_size 512
