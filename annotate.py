#!/home/caetano/venv/bin python

import cv2
import pandas as pd
import os
import numpy as np


RESET_LABELS = False

input_file  = 'flood_images.csv'
dataset_path = '/media/caetano/Caetano/enoe'
#output_file = input('Name of the annotated csv file (output):')


df = pd.read_csv(input_file, parse_dates=['datetime'], index_col=0)

if RESET_LABELS or not 'level' in df.columns:
    df['level'] = np.nan
df['level'] = pd.to_numeric(df['level'])

flood1 = df[ (df['datetime'] > pd.to_datetime('2018-11-01')) &
             (df['datetime'] < pd.to_datetime('2019-03-01')) ]
flood2 = df[ (df['datetime'] > pd.to_datetime('2019-11-01')) &
             (df['datetime'] < pd.to_datetime('2020-02-01')) ]
flood3 = df[ (df['datetime'] > pd.to_datetime('2020-11-01')) &
             (df['datetime'] < pd.to_datetime('2021-03-01')) ]
flood4 = df[ (df['datetime'] > pd.to_datetime('2021-11-01')) &
             (df['datetime'] < pd.to_datetime('2022-03-01')) ]

subset = pd.concat([flood1,flood2,flood3,flood4])
subset[subset['place'].isna()] = 'unknown'
subset = subset.sort_values(by=['place','datetime'], ascending=[True,True])
print(subset.head())
subset = subset[subset['level'].isna()]
print(subset.head())

print('1 - low\n\
       2 - mid\n\
       3 - high\n\
       4 - flood\n\
       0 - invalid image\n\
       c - commit\n\
       q - quit')

last_labels = list()
last_indexes = list()
for index, row in subset.iterrows():
    print(row['path'])
    img = cv2.imread(os.path.join(dataset_path,row['path']))
    cv2.imshow('img', img)
    key = ''
    while not key in ['q', '0', '1', '2', '3', '4']:
        key = chr(cv2.waitKey())
        if key=='c':
            for idx, lbl in zip(last_indexes,last_labels):
                df.loc[idx,['level']] = [lbl]
            #for i, lbl in enumerate(last_labels):
            #    df.loc[index-len(last_labels)+1+i,['level']] = [lbl]
            last_labels = list()
            last_indexes = list()
            print('Commited values')
    if key == 'q':
        break
    else:
        last_labels.append(key)
        last_indexes.append(index)
        print(f'Level {key}')

print(df[~df['level'].isna()])
df.to_csv('flood_images.csv')