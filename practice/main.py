import pandas as pd
import numpy as np

def split_data(data=None, test_size=0.3):
    if not data:
        return None, None
    msk = np.random.rand(len(data)) < (1 - test_size)
    return data[msk], data[~msk]

def evaluate(model=None, data=None, features=[], run=3):
    scores = []
    for i in range(run):
        train_data, test_data = split_data(data, 0.3)



data = pd.read_csv('data.csv')
