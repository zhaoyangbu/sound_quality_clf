import pandas as pd
import numpy as np
import sys
import requests
import os
import warnings
import time
warnings.filterwarnings("ignore")


def gen_df(filename):
    df = pd.read_excel(filename)
    new_df = pd.DataFrame(df, columns=['scdid','url'])
    return new_df
            


def download_url(url,id):
    start = time.time()
    resp = requests.get(url, timeout=300, verify=False)
    res = resp.content
    with open("songs_infer/{}.mp3".format(id),"wb") as file:
        file.write(res)
        end = time.time()
        duration = end-start
        print("{}.mp3 downloaded successfully, takes {}s".format(id, duration))



def run(df):
    num = len(df['url'])
    for i in range(num):
        if os.path.exists("songs_infer/{}.mp3".format(df['scdid'][i])):
            print("songs_infer/{}.mp3 is already been downloaded before".format(df['scdid'][i]))
        else:
            download_url(df['url'][i], df['scdid'][i])



if __name__ == "__main__":
    file = sys.argv[1]
    df = gen_df('{}'.format(file))
    run(df)
