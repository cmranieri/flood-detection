#!/home/caetano/venv/bin python

import cv2
import pandas as pd
import os

input_file  = 'flood_images.csv'
dataset_path = '/home/caetano/enoe'
#output_file = input('Name of the annotated csv file (output):')

df = pd.read_csv(input_file)
subset = df.sample(n=10)
for row in subset['path']:
    print(row)
    img = cv2.imread(os.path.join(dataset_path,row))
    cv2.imshow('img', img)
    label = int(cv2.waitKey())