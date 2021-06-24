import io
import os
import requests

import pandas  as pd


def read_uci(url, directory, sep=" "):
 
    import warnings
    warnings.filterwarnings("ignore")

    s=requests.get(url, verify=False).content
    data=pd.read_csv(io.StringIO(s.decode('utf-8')), sep, engine='python', header=None)
    
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    return data