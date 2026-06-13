########################################################
# CUDA_VISIBLE_DEVICES=1 \
python train_vae.py \
--data_root './Data/datasets/.csv' \
--dataset sine \
--train_epochs 15001 \
--seq_length 24 \
--feat_dim 5 \
--D_TOKEN 10 \
--N_HEAD 1 \
--FACTOR 16 \
--NUM_LAYERS 2 \
--batch_size 1024 

# CUDA_VISIBLE_DEVICES=1 \
python train_ldm.py \
--dataset sine \
--num_epochs 15001 \
--seq_length 24 \
--feat_dim 5 \
--D_TOKEN 10 \
--N_HEAD 1 \
--FACTOR 16 \
--NUM_LAYERS 2 \
--LDM_dim 2048 \
--batch_size 1024 
