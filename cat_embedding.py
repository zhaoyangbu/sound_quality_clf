import pandas as pd
import sys

def cat_embedding(new_path):
    df1 = pd.read_json('embeddings.json')
    df2 = pd.read_json('{}/embeddings.json'.format(new_path))
    frames = [df1, df2]
    new_df = pd.concat(frames, ignore_index=True)

    new_df.to_json('embeddings.json')
    
if __name__ == "__main__":
    path = sys.argv[1]
    cat_embedding(path)

# $python cat_embedding.py 'new_embd_path'