import pandas as pd
import numpy as np


def get_adl(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    df = pd.DataFrame(columns=['event','timestamp'])
    
    for l in lines[2:]: #ignore header and spacer line
        l = l.strip().split("\t")
        l = list(filter(lambda x:x !='', l))
        new_row_start = {'event': l[2] + "_start", 'timestamp': l[0].strip()}
        new_row_end = {'event': l[2] + "_end", 'timestamp': l[1].strip()}
        new_df = pd.DataFrame([new_row_start, new_row_end], index=[0,1])
        df = pd.concat([df, new_df])
    
    cat_codes = df["event"].astype('category')
    df['event'] = cat_codes.cat.codes

    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64)  // 10**9
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    df.reset_index(drop=True, inplace=True)
    
    return df,cat_codes