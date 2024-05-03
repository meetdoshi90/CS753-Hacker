# AVRelScore: test example on LRS2
python test.py \
    --data_path '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/lrs2_processed' \
    --data_type LRS2 \
    --model_conf ./src/models/model.json \
    --split_file ./src/data/LRS2/test.ref \
    --checkpoint '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/ft_checkpoints_new/Best_0000_02000_0.21.ckpt' \
    --architecture AVRelScore \
    --results_path './test_results_fusion.txt' \
    --rnnlm ./checkpoints/LM/lm_en_lrs2/model.pth \
    --rnnlm_conf ./checkpoints/LM/lm_en_lrs2/model.json \
    --beam_size 40 \
    --ctc_weight 0.1 \
    --lm_weight 0.5 \
    --gpu 3