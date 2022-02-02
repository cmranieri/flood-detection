#!/home/caetano/venv/bin python

import cv2
import pandas as pd
import os
import numpy as np


input_file  = 'flood_images.csv'
dataset_path = '/media/caetano/Caetano/enoe'
#output_file = input('Name of the annotated csv file (output):')


df = pd.read_csv(input_file, parse_dates=['datetime'], index_col=0)
print(df.head())

if not 'level' in df.columns:
    df['level'] = np.nan

flood1 = df[ (df['datetime'] > pd.to_datetime('2018-11-01')) &
             (df['datetime'] < pd.to_datetime('2019-02-01')) ]
flood2 = df[ (df['datetime'] > pd.to_datetime('2019-11-01')) &
             (df['datetime'] < pd.to_datetime('2020-02-01')) ]
flood3 = df[ (df['datetime'] > pd.to_datetime('2020-11-01')) &
             (df['datetime'] < pd.to_datetime('2021-02-01')) ]
flood4 = df[ (df['datetime'] > pd.to_datetime('2021-11-01')) &
             (df['datetime'] < pd.to_datetime('2022-02-01')) ]

subset = pd.concat([flood1,flood2,flood3,flood4])
subset = subset[subset['level'].isna()]
subset = subset.sort_values(by=['place','datetime'], ascending=[True,True])

print('1 - low\n2 - mid\n3 - high\n4 - flood\nq - quit')
for index, row in subset.iterrows():
    print(row['path'])
    img = cv2.imread(os.path.join(dataset_path,row['path']))
    cv2.imshow('img', img)
    key = chr(cv2.waitKey(0))
    while not key in ['q', '1', '2', '3', '4']:
        print('Invalid key!')
        key = chr(cv2.waitKey())
    if key=='q':
        break
    elif key=='n':
        continue
    df.loc[index,['level']] = [key]
    print(f'Registered level {key}!')

print(df[~df['level'].isna()])
df.to_csv('flood_images.csv')