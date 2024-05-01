# Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring
 Joanna Hong, Minsu Kim, Jeongsoo Choi, Yong Man Ro

CS753 Hacker implementation

Team Members:

22m0742 Meet Doshi

22m2119 Badri Vishal Kasuba

22m0816 Palash Dilip Moon

22m0764 Sankara Sri Raghava



### Overall ideas implemented or tried:
- Using extra Audio-Visual data with synthetic transcripts 
- Comparison of using both occlusion patches and blur vs using only blur.
- Fusion loss term using noise labels.


Code primarily taken from three repositories

- https://github.com/joannahong/AV-RelScore (Official implementation)
- https://github.com/ms-dot-k/AVSR (Open resource reimplementation)
- https://github.com/mpc001/auto_avsr (Used to generated synthetic transcription for voxceleb)

Download landmarks and noise from: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
> Follow README in AVSR/ folder

Steps:

1. Downlaod voxceleb2 from huggingface https://huggingface.co/datasets/ProgramComputer/voxceleb (Unofficial implementation) into the voxceleb folder and also download voxceleb landmarks following readme in auto_avsr/ folder
2. Use asr_infer.py to generate transcriptions. (Follow bash scripts)
3. Create aud_dir, txt_dir, vid_dir for audio video text respectively. 
4. Follow readme in AVSR Folder to install lrs2. Download lrs2 landmarks and put it inside AVSR/lrs2 folder.
5. Other scripts are same as authors
6. Use AVSR/vox_preprocessing.py to process vox in same format as LRS2.
7. Use run.sh to test the model and use run_ft.sh to fine tune the existing model by the authors.


Feel free to contact at: meetdoshi@cse.iitb.ac.in
