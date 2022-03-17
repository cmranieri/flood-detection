#!/home/caetano/venv/bin python

import cv2
import pandas as pd
import os
import numpy as np


RESET_LABELS = False

csv_path  = '../resources/flood_images_annot.csv'
dataset_path = '/home/caetano/enoe2'


def checkpoint_values(df, last_labels, last_indexes):
    for idx, lbl in zip(last_indexes,last_labels):
        df.loc[idx,['level']] = [lbl]
    return df


df = pd.read_csv(csv_path, parse_dates=['datetime'], index_col=0)

if RESET_LABELS or not 'level' in df.columns:
    df['level'] = np.nan
df['level'] = pd.to_numeric( df['level'] )

flood1 = df[ (df['datetime'] > pd.to_datetime('2018-11-01')) &
             (df['datetime'] < pd.to_datetime('2019-03-01')) ]
flood2 = df[ (df['datetime'] > pd.to_datetime('2019-11-01')) &
             (df['datetime'] < pd.to_datetime('2020-02-01')) ]
flood3 = df[ (df['datetime'] > pd.to_datetime('2020-11-01')) &
             (df['datetime'] < pd.to_datetime('2021-03-01')) ]
flood4 = df[ (df['datetime'] > pd.to_datetime('2021-11-01')) &
             (df['datetime'] < pd.to_datetime('2022-03-01')) ]

subset = pd.concat([flood1,flood2,flood3,flood4])
mask = subset['place'].isna()
subset.loc[mask,'place'] = 'unknown'
subset = subset.sort_values( by=['place','datetime'], ascending=[True,True] )
print(subset.head())
subset = subset[ ~subset['place'].str.contains('SESC')]
subset = subset[subset['level'].isna()]
#subset = subset[subset['level']==3]
print(subset.head())

print('1 - low\n\
       2 - mid\n\
       3 - high\n\
       4 - flood\n\
       0 - not visible\n\
       5 - invalid image\n\
       c - set checkpoint\n\
       q - save last checkpoint and quit')

last_labels = list()
last_indexes = list()
quit_command = False
for index, row in subset.iterrows():
    print(row['path'])
    img = cv2.imread(os.path.join(dataset_path,row['path']))
    if img is None:
        continue
    cv2.imshow('img', img)
    key = ''
    while not key in ['q', '0', '1', '2', '3', '4', '5']:
        key = chr(cv2.waitKey())
        if key=='c':
            df = checkpoint_values(df, last_labels, last_indexes)
            print('Checkpointed values')
            last_labels = list()
            last_indexes = list()
    if key == 'q':
        quit_command = True
        break
    elif key == '5':
        continue
    else:
        last_labels.append(key)
        last_indexes.append(index)
        print(f'Level {key}')
# If finished annotating
if not quit_command:
    df = checkpoint_values(df, last_labels, last_indexes)

#print(df[ (~df['level'].isna()) & df['place'].str.contains('SHOP') ])
print(df[ ~df['level'].isna() ])
df.to_csv(csv_path)
