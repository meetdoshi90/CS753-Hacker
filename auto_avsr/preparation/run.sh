# screen -L -Logfile "screenlog.vox-process-$i" -S "meet-vox-process-$i" -d -m python preprocess_vox2.py \
#     --vid-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/vid_dir" \
#     --aud-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/aud_dir" \
#     --landmarks-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/vox2_landmarks" \
#     --detector "retinaface" \
#     --root-dir "./vox_processed/" \
#     --groups 128 \
#     --job-index $i
for i in {0..0}
do
    screen -L -Logfile "screenlog.vox-process-$i" -S "meet-vox-process-$i" -d -m python preprocess_vox2.py \
        --vid-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/vid_dir" \
        --aud-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/aud_dir" \
        --landmarks-dir "/raid/nlp/pranavg/meet/ASR/Hacker/voxceleb2/vox2_landmarks" \
        --detector "retinaface" \
        --root-dir "./vox_processed/" \
        --groups 128 \
        --job-index $i
    echo $i
done