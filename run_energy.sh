########################################################
# CUDA_VISIBLE_DEVICES=1 \
python train_vae.py \
--data_root './Data/datasets/energy_data.csv' \
--dataset energy \
--train_epochs 15001 \
--seq_length 24 \
--feat_dim 28 \
--D_TOKEN 56 \
--N_HEAD 1 \
--FACTOR 16 \
--NUM_LAYERS 3 \
--batch_size 2048 

# CUDA_VISIBLE_DEVICES=1 \
python train_ldm.py \
--dataset energy \
--num_epochs 15001 \
--seq_length 24 \
--feat_dim 28 \
--D_TOKEN 56 \
--N_HEAD 1 \
--FACTOR 16 \
--NUM_LAYERS 3 \
--LDM_dim 6144 \
--batch_size 2048



