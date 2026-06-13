########################################################
CUDA_VISIBLE_DEVICES=0 \
python train_vae.py \
--data_root './Data/datasets/ETTh.csv' \
--dataset etth \
--train_epochs 15 \
--seq_length 24 \
--feat_dim 7 \
--D_TOKEN 14 \
--N_HEAD 2 \
--FACTOR 16 \
--NUM_LAYERS 3 \
--batch_size 1024 

python train_ldm.py \
--dataset etth \
--num_epochs 15 \
--seq_length 24 \
--feat_dim 7 \
--D_TOKEN 14 \
--N_HEAD 2 \
--FACTOR 16 \
--NUM_LAYERS 3 \
--LDM_dim 1024 \
--batch_size 1024 
