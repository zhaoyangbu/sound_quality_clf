import pandas as pd
import numpy as np
import sys
import requests
import os
import warnings
warnings.filterwarnings("ignore")


def gen_df(filename):
    df = pd.read_excel(filename)
    new_df = pd.DataFrame(df, columns=['资源编号','URL','是否通过'])
    return new_df
            


def download_url(url,id,is_pass):
    resp = requests.get(url, timeout=300, verify=False)
    res = resp.content
    with open("songs/{}_{}.mp3".format(id, is_pass),"wb") as file:
        file.write(res)
        print("{}_{}.mp3 downloaded successfully".format(id, is_pass))



def run(df):
    num = len(df['URL'])
    for i in range(num):
        if df['是否通过'][i] == '通过':
            is_pass = 1
        else:   
            is_pass = 0
        if os.path.exists("songs/{}_{}.mp3".format(df['资源编号'][i], is_pass)):
            print("songs/{}_{}.mp3 is already been downloaded before".format(df['资源编号'][i], is_pass))
        else:
            download_url(df['URL'][i], df['资源编号'][i],  is_pass)



if __name__ == "__main__":
    file = sys.argv[1]
    df = gen_df('{}'.format(file))
    run(df)

# Use: % python gen_dataset.py 'file'

