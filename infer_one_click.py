import pandas as pd
import numpy as np
import sys
from sys import path
import requests
import os
import warnings
import time
from sklearn.linear_model import LogisticRegression 
warnings.filterwarnings("ignore")
import librosa
from panns_inference import SoundEventDetection, labels, AudioTagging
import pickle



# Generate inference dataset(audio file)
def get_df(filename):
    df = pd.read_excel(filename)
    new_df = pd.DataFrame(df, columns=['scdid','url'])
    return new_df
            


def download_url(url,id):
    start = time.time()
    resp = requests.get(url, timeout=300, verify=False)
    res = resp.content
    with open("inference_dataset/{}.ogg".format(id),"wb") as file:
        file.write(res)
        end = time.time()
        duration = end-start
        print("{}.ogg downloaded successfully, takes {}s".format(id, duration))



# Audio to embedding
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
        file_size = os.path.getsize(path+'/'+file)
        if file_size <= 150000:
            continue
        else: 
            start = time.time()
            if file[-3:] != 'ogg':  
                continue
            else:
                f = file.split('.')
                id = f[0]
                embd = get_embedding(path+'/'+file)
                temp_list = [id, embd]
                embd_list.append(temp_list)
                end = time.time()
                duration = end-start
                print('takes {}s to embed id:{}'.format(duration, id))
    return embd_list



# Embeddings to result
def gen_result_list(df, model):
    new_list_pass = []
    new_list_fail = []
    for i in range(len(df)):
        temp_list = []
        id = df['id'][i]
        result = model.predict([df['embedding'][i]])
        if result == 0:
            temp_list_fail = [id, result]
            new_list_fail.append(temp_list_fail)
        else:
            temp_list_pass = [id, result]
            new_list_pass.append(temp_list_pass)
    return new_list_pass, new_list_fail





#  State transition
def url2ogg(df):
    num = len(df['url'])
    for i in range(num):
        if os.path.exists("inference_dataset/{}.ogg".format(df['scdid'][i])):
            print("inference_dataset/{}.ogg is already been downloaded before".format(df['scdid'][i]))
        else:
            download_url(df['url'][i], df['scdid'][i])


def ogg2result(path, save_embed=False):
    embedding_list = get_embd_list(path)
    df = pd.DataFrame(embedding_list, columns=['id' ,'embedding'])
    if save_embed == False:
        pass
    else:
        df.to_json('results/infer_embeddings.json')
    model = pickle.load(open('models/lr_model.sav', 'rb'))
    new_list_pass, new_list_fail = gen_result_list(df, model)

    df2 = pd.DataFrame(new_list_pass, columns=['id', 'label'])
    df2.to_excel('results/infer_result_pass.xls')

    df3 = pd.DataFrame(new_list_fail, columns=['id', 'label'])
    df3.to_excel('results/infer_result_fail.xls')

if __name__ == "__main__":
    file = sys.argv[1]
    df = get_df('{}'.format(file))
    url2ogg(df)
    ogg2result('inference_dataset', save_embed=False)


# python infer_one_click.py "url.xls"