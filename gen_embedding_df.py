import os
import sys
from sys import path
import librosa
import numpy as np
import pandas as pd
from panns_inference import SoundEventDetection, labels, AudioTagging
at = AudioTagging(checkpoint_path=None, device='cuda')


def get_embedding(path):
    audio, _ = librosa.core.load(path, sr=32000, mono=True)
    audio = audio[None, :]
    _, embedding = at.inference(audio)
    embedding = embedding/np.linalg.norm(embedding)
    embedding = embedding.tolist()[0]
    return embedding


def get_embd_list(path):
    files = os.listdir(path)
    embd_list = []
    for file in files:
        if file[-3:] != 'mp3':
            continue
        else:
            f = file.split('_')
            id = f[0]

            label = f[1].split('.')[0]
            embd = get_embedding(path+'/'+file)
            temp_list = [id, embd, label]
            embd_list.append(temp_list)
    return embd_list

def run(path):
    embedding_list = get_embd_list(path)
    df = pd.DataFrame(embedding_list, columns=['id' ,'embedding', 'label'])
    df.to_json('{}/embeddings.json'.format(path))

if __name__ == "__main__":
    path = sys.argv[1]
    run(path)
    

# Use: % python gen_embedding_df.py 'path'





