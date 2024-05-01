# # AVRelScore: Distributed training example using 2 GPUs on LRS2 (nproc_per_node should have the same number with gpus)
# python train.py \
#     --data_path '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/lrs2_processed' \
#     --data_type LRS2 \
#     --split_file ./src/data/LRS2/0_600.txt \
#     --model_conf ./src/models/model.json \
#     --checkpoint_dir '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/ft_checkpoints_new' \
#     --checkpoint '/raid/nlp/pranavg/meet/ASR/Hacker/AVRelScore_LRS2.ckpt' \
#     --v_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar \
#     --a_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar \
#     --batch_size 6 \
#     --update_frequency 1 \
#     --epochs 5 \
#     --eval_step 1000 \
#     --visual_corruption \
#     --architecture AVRelScore \
#     --gpu 2
# AVRelScore: Distributed training example using 2 GPUs on LRS2 (nproc_per_node should have the same number with gpus)
python train.py \
    --data_path '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/vox_processed' \
    --data_type voxceleb \
    --split_file ./src/data/voxceleb/split_file.txt \
    --model_conf ./src/models/model.json \
    --checkpoint_dir '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/ft_checkpoints_vox' \
    --checkpoint '/raid/nlp/pranavg/meet/ASR/Hacker/AVRelScore_LRS2.ckpt' \
    --v_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar \
    --a_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar \
    --batch_size 6 \
    --update_frequency 1 \
    --epochs 5 \
    --eval_step 1000 \
    --visual_corruption \
    --architecture AVRelScore \
    --gpu 0
