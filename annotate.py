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

flood1 = df[ (df['datetime'] > pd.to_datetime('2020-12-25')) &
             (df['datetime'] < pd.to_datetime('2020-12-28')) ]

subset = flood1[flood1['level'].isna()]

print('1 - low\n2 - mid\n3 - high\n4 - flood\nn - unreadable\nq - quit')
for index, row in subset.iterrows():
    print(row['path'])
    img = cv2.imread(os.path.join(dataset_path,row['path']))
    cv2.imshow('img', img)
    key = chr(cv2.waitKey(0))
    while not key in ['q', 'n', '1', '2', '3', '4']:
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