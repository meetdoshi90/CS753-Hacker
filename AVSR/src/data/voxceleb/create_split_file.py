import os

path = '/raid/nlp/pranavg/meet/ASR/Hacker/AVSR/vox_processed/Text/dev/'

split_file = []
val_file = []

folders = os.listdir(path)

for folder in folders:
    subfolders = os.listdir(path+folder)
    for subfolder in subfolders:
        files = os.listdir(path+folder+'/'+subfolder)
        split_file_names = ['dev/'+folder+'/'+subfolder+'/'+x[:-4] for x in files]
        val_file_names = [folder+'/'+subfolder+'/'+x[:-4] for x in files]
        split_file.extend(split_file_names)
        val_file.extend(val_file_names)

with open('./split_file.txt', 'w') as f:
    f.writelines('\n'.join(split_file))

val_file = val_file[:1000]

with open('./val.txt', 'w') as f:
    f.writelines('\n'.join(val_file))

