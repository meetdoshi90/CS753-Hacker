CUDA_VISIBLE_DEVICES=3
for i in {24..31}
do
    screen -L -Logfile "screenlog.whisper-$i" -S "meet-whisper-$i" -d -m python3 asr_inference.py \
        --audio-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/aud_dir/" \
        --groups 32 \
        --job-index $i \
        --device $CUDA_VISIBLE_DEVICES
done
#, device=f'cuda:{str(args.device)}'