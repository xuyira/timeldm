########################################################
# CUDA_VISIBLE_DEVICES=0 \
python train_vae.py \
--data_root './Data/datasets/fMRI/' \
--dataset fmri \
--train_epochs 15001 \
--seq_length 24 \
--feat_dim 50 \
--D_TOKEN 100 \
--N_HEAD 1 \
--FACTOR 16 \
--NUM_LAYERS 2 \
--batch_size 1024 


python train_ldm.py \
--dataset fmri \
--num_epochs 15001 \
--seq_length 24 \
--feat_dim 50 \
--D_TOKEN 100 \
--N_HEAD 1 \
--FACTOR 16 \
--NUM_LAYERS 2 \
--LDM_dim 4096 \
--batch_size 1024 



