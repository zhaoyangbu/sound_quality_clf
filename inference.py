import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse.construct import random
from sklearn.linear_model import LogisticRegression 
import pickle





def gen_result_list(df, model):
    new_list = []
    for i in range(len(df)):
        temp_list = []
        id = df['id'][i]
        result = model.predict([df['embedding'][i]])
        temp_list = [id, result]
        new_list.append(temp_list)
    return new_list

def run():
    model = pickle.load(open('lr_model.sav', 'rb'))
    df = pd.read_json('songs_infer/infer_embeddings.json')
    new_list = gen_result_list(df, model)

    df2 = pd.DataFrame(new_list, columns=['id', 'label'])
    df2.to_excel('infer_result.xls')

if __name__ == "__main__":
    run()
    

