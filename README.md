# Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring
 Joanna Hong, Minsu Kim, Jeongsoo Choi, Yong Man Ro

CS753 Hacker implementation

Team Members:

22m0742 Meet Doshi

22m2119 Badri Vishal Kasuba

22m0816 Palash Dilip Moon

22m0764 Sankara Sri Raghava

### Overall ideas implemented or tried:
(Not given in the paper but we explored it on our own intuition)

- Using extra Audio-Visual data with synthetic transcripts (gave initial promising results)
- Comparison of using both occlusion patches and blur vs using only blur. (almost gave similar results, might have worked if done during pretraining)
- Fusion loss term using noise labels. (We use this loss as an explicit signal to tell the model to guide the AVRelScore away from unreliable video frames. We implement this only for video frames where noise is present but we can do it for audio frames as well. We do this by also returning a noise label vector in the data loader which adds the noise to the video frame and add a BCE loss which we call fusion loss to allow the AVRelScore to be more reliable for occlusion frames as well.) We do this because as you can see below AVRelScore gives high reliability scores in frames with occlusion objects as well. We understand sometimes it may be reliable but an explicit signal may help. Changes made at "./AVSR/espnet/nets/pytorch_backend/e2e_asr_transformer.py"




Warning! : The datasets are too large to download. Details to download are given below

To download
- voxceleb2 mp4 and m4a files (convert m4a to wav)
- voxceleb2 landmarks
- lrs2 mp4 files (audio is already present)
- lrs2 landmarks
- pretrained backbones for audio and video
- occulusion noise images
- audio noise file

## Start with installing requirements

> pip3 install -r requirements.text

## Pre-processing VoxCeleb2

To pre-process the VoxCeleb2 dataset, please follow these steps:

1. Download the VoxCeleb2 dataset from the official website.
    https://huggingface.co/datasets/ProgramComputer/voxceleb  
    
    Inside the vox2 folder

    > vox2_dev_aac_partaa - vox2_dev_aac_partah 83GB

    > vox2_dev_mp4_partaa - vox2_dev_mp4_partai 255GB

2. Download pre-computed landmarks below. Once you've finished downloading the five files, simply merge them into one single file using `zip -FF vox2_landmarks.zip --out single.zip`, and then decompress it. If you leave `landmarks-dir` empty, landmarks will be provided with the used of `detector`.

    | File Name              | Source URL                                                                        | File Size |
    |------------------------|-----------------------------------------------------------------------------------|-----------|
    | vox2_landmarks.zip     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.zip)     | 18GB      |
    | vox2_landmarks.z01     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z01)     | 20GB      |
    | vox2_landmarks.z02     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z02)     | 20GB      |
    | vox2_landmarks.z03     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z03)     | 20GB      |
    | vox2_landmarks.z04     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z04)     | 20GB      |

3. Since voxceleb just has audio and video separately, but audio files are in m4a format, you need to convert them to wav using the script in ./voxceleb2/aud_dir/mconvert.py . This will create files with same name in the aud_dir folder of voxceleb in wav format.
    ```Shell 
    python3 voxceleb2/aud_dir/convert.py
    ```


4. Now you have the aud_dir and vid_dir ready along with vox2 landmarks. Now you also need text transcripts for those audio files. So now we will use whisper to generate large scale synthetic transcripts. To generate synthetic transcripts we divide the audio files to different segments to run them in parallel. Look at the script at ./voxceleb2/whisper_run.sh which generates 8 processes per GPU. You can run it 4 times (8*4) to get all 32 segments done in parallel. 
    ```Shell 
    bash ./voxceleb2/whisper_run.sh
    ```

4. Run the following command to pre-process dataset (update the paths in the script):
    ```Shell 
    bash ./AVSR/process-vox.sh
    ```
We suppose the data directory is constructed as

```
Voxceleb
├── txt_dir
|   ├── dev
|   |   ├── *
|   |   |   ├── *
|   |   |   |   └── *.txt
├── aud_dir
|   ├── dev
|   |   ├── *
|   |   |   ├── *
|   |   |   |   └── *.txt
├── vid_dir
|   ├── dev
|   |   ├── *
|   |   |   ├── *
|   |   |   |   └── *.txt
```

## Preparation of LRS2
### Dataset Download
LRS2/LRS3 dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/
Script to download with credentials can be found at:
    > bash ./AVSR/lrs2/down.sh

### Landmark Download
For data preprocessing, download the landmark of LRS2 and LRS3 from the [repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#Model-Zoo). 
(Landmarks for "VSR for multiple languages models")

### Occlusion Data Download
For visual corruption modeling, download `coco_object.7z` from the [repository](https://github.com/kennyvoo/face-occlusion-generation). 

Unzip and put the files at
```
./AVSR/occlusion_patch/object_image_sr
./AVSR/occlusion_patch/object_mask_x4
```

### Babble Noise Download
For audio corruption modeling, download babble noise file from [here](https://drive.google.com/file/d/15CSWCYz12CIsgFeDT139CiCc545jhbyK/view?usp=sharing). 

put the file at
```
./AVSR/src/data/babbleNoise_resample_16K.npy
```

### Pre-trained Frontends 
For initializing visual frontend and audio frontend, please download the pre-trained models from the [repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo). (resnet18_dctcn_audio/resnet18_dctcn_video)

Put the .tar file at
```
./AVSR/checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar
./AVSR/checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar
```

### Preprocessing LRS2
After download the dataset and landmark, we 1) align and crop the lip centered video, 2) extract audio, 3) obtain aligned landmark.
We suppose the data directory is constructed as
```
LRS2
├── main
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
├── pretrain
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
```

Run preprocessing with the following commands:
```shell
# For LRS2
python ./AVSR/preprocessing.py \
--data_path '/path_to/LRS2' \
--data_type LRS2 \
--landmark_path '/path_to/LRS2_landmarks' \
--save_path '/path_to/LRS2_processed' 
```
> Example can be found at ./AVSR/process.sh

## Training/Finetuning the Model
Basically, you can choice model architecture with the parameter `architecture`. <br>
There are three options for the `architecture`: `AVRelScore` but only this is working. <br>
To train the model, run following command:

```shell
# AVRelScore: Distributed training example using 2 GPUs on LRS2 (nproc_per_node should have the same number with gpus)
python train.py \
    --data_path './AVSR/vox_processed' \
    --data_type voxceleb \
    --split_file ./AVSR/src/data/voxceleb/split_file.txt \
    --model_conf ./AVSR/src/models/model.json \
    --checkpoint_dir './AVSR/ft_checkpoints_vox' \
    --checkpoint './AVSR/AVRelScore_LRS2.ckpt' \
    --v_frontend_checkpoint ./AVSR/checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar \
    --a_frontend_checkpoint ./AVSR/checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar \
    --batch_size 6 \
    --update_frequency 1 \
    --epochs 5 \
    --eval_step 1000 \
    --visual_corruption \
    --architecture AVRelScore \
    --gpu 0
```


Descriptions of training parameters are as follows:
- `--data_path`: Preprocessed Dataset location (LRS2 or LRS3)
- `--data_type`: Choose to train on LRS2 or LRS3
- `--split_file`: train and validation file lists (you can do curriculum learning by changing the split_file, 0_100.txt consists of files with frames between 0 to 100; training directly on 0_600.txt is also not too bad.)
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint`: saved checkpoint where the training is resumed from
- `--model_conf`: model_configuration
- `--wandb_project`: if want to use wandb, please set the project name here. 
- `--batch_size`: batch size
- `--update_frequency`: update_frquency, if you use too small batch_size increase update_frequency. Training batch_size = batch_size * udpate_frequency
- `--epochs`: number of epochs
- `--tot_iters`: if set, the train is finished at the total iterations set
- `--eval_step`: every step for performing evaluation
- `--fast_validate`: if set, validation is performed for a subset of validation data
- `--visual_corruption`: if set, we apply visual corruption modeling during training
- `--architecture`: choose which architecture will be trained. (options: AVRelScore, VCAFE, Conformer)
- `--gpu`: gpu number for training
- `--distributed`: if set, distributed training is performed
- Refer to `train.py` for the other training parameters

## Testing the Model
To test the model, run following command:
```shell
# AVRelScore: test example on LRS2
python test.py \
    --data_path './AVSR/lrs2_processed' \
    --data_type LRS2 \
    --model_conf ./src/models/model.json \
    --split_file ./src/data/LRS2/test.ref \
    --checkpoint './AVSR/ft_checkpoints_new/Best_0000_02000_0.21.ckpt' \
    --architecture AVRelScore \
    --results_path './AVSR/test_results_fusion.txt' \
    --rnnlm ./AVSR/checkpoints/LM/lm_en_lrs2/model.pth \
    --rnnlm_conf ./AVSR/checkpoints/LM/lm_en_lrs2/model.json \
    --beam_size 40 \
    --ctc_weight 0.1 \
    --lm_weight 0.5 \
    --gpu 3
```



Descriptions of testing parameters are as follows:
- `--data_path`: Preprocessed Dataset location (LRS2 or LRS3)
- `--data_type`: Choose to train on LRS2 or LRS3
- `--split_file`: set to test.ref (./src/data/LRS2./test.ref or ./src/data/LRS3/test.ref)
- `--checkpoint`: model for testing
- `--model_conf`: model_configuration
- `--architecture`: choose which architecture will be trained. (options: AVRelScore, VCAFE, Conformer)
- `--gpu`: gpu number for training
- `--rnnlm`: language model checkpoint
- `--rnnlm_conf`: language model configuration
- `--beam_size`: beam size
- `--ctc_weight`: ctc weight for joint decoding
- `--lm_weight`: language model weight for decoding
- Refer to `test.py` for the other parameters

## Pre-trained model checkpoints
Download the pre-trained AVSR models (VCAFE and AVRelScore) on LRS2 and LRS3 datasbases. (Below WERs can be obtained at `beam_width`: 40, `ctc_weight`: 0.1, `lm_weight`: 0.5) 

| Model |       Dataset       |   WER   |
|:-------------------:|:-------------------:|:--------:|
|VCAFE|LRS2 |   [4.459](https://drive.google.com/file/d/1509xCvaMgMwtfJxE04zPgWRgiYgy3fFD/view?usp=sharing)  |
|VCAFE|LRS3 |   [2.821](https://drive.google.com/file/d/1539Td4FaBCta-1KOCKEz1QovAlzx-DkO/view?usp=sharing)  |
|AVRelScore|LRS2 |   [4.129](https://drive.google.com/file/d/157fsllT8pldpuCFtVTRYUNd1yK5AoE-b/view?usp=sharing)  |
|AVRelScore|LRS3 |   [2.770](https://drive.google.com/file/d/159gCUDJAKDIYchS5iNXQ1M5pHnPGIIDd/view?usp=sharing)  |

You can find the pre-trained Language Model in the following [repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#Model-Zoo).
Put the language model at
```
./AVSR/checkpoints/LM/model.pth
./AVSR/checkpoints/LM/model.json
```

## Acknowledgment
The code are based on the following two repositories,
Code primarily taken from three repositories

- https://github.com/joannahong/AV-RelScore (Official implementation)
- https://github.com/ms-dot-k/AVSR (Open resource reimplementation)
- https://github.com/mpc001/auto_avsr (Used to generated synthetic transcription for voxceleb)

Download landmarks and noise from: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
> Follow README in AVSR/ folder

Summary of steps:

1. Downlaod voxceleb2 from huggingface https://huggingface.co/datasets/ProgramComputer/voxceleb (Unofficial implementation) into the voxceleb folder and also download voxceleb landmarks following readme in auto_avsr/ folder
2. Use asr_infer.py to generate transcriptions. (Follow bash scripts)
3. Create aud_dir, txt_dir, vid_dir for audio video text respectively. 
4. Follow readme in AVSR Folder to install lrs2. Download lrs2 landmarks and put it inside AVSR/lrs2 folder.
5. Other scripts are same as authors
6. Use AVSR/vox_preprocessing.py to process vox in same format as LRS2.
7. Use run.sh to test the model and use run_ft.sh to fine tune the existing model by the authors.

## Citation
If you find this work useful in your research, please cite the papers:
```
@inproceedings{hong2023watch,
  title={Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring},
  author={Hong, Joanna and Kim, Minsu and Choi, Jeongsoo and Ro, Yong Man},
  booktitle={Proc. CVPR},
  pages={18783--18794},
  year={2023}
}
```
```
@inproceedings{hong2022visual,
  title={Visual Context-driven Audio Feature Enhancement for Robust End-to-End Audio-Visual Speech Recognition},
  author={Hong, Joanna and Kim, Minsu and Ro, Yong Man},
  booktitle={Proc. Interspeech},
  pages={2838--2842},
  year={2022},
  organization={ISCA}
}
```

```
@inproceedings{Ma_2023,
   title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
   url={http://dx.doi.org/10.1109/ICASSP49357.2023.10096889},
   DOI={10.1109/icassp49357.2023.10096889},
   booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   publisher={IEEE},
   author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
   year={2023},
   month=jun }
```

Feel free to contact at: meetdoshi@cse.iitb.ac.in
