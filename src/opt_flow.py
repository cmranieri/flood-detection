import cv2
import os
import numpy as np
import pandas as pd
import enoe_utils


def compute_flow_pair(frame1_gray, frame2_gray):
    # Computes dense optical flow using the Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, 
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def flow2img(flow):
    flow_img = np.zeros_like(flow)
    cv2.normalize(flow, flow_img, 0, 255, cv2.NORM_MINMAX)
    return flow_img


if __name__=='__main__':
    enoe_dir = '/enoe'
    csv_path = '../resources/flood_images_annot.csv'
    csv_flow_path = '../resources/flood_flow_annot.csv'
    flows_dir = '/flow'
    max_time_diff = 20
    os.makedirs(flows_dir, exist_ok=True)

    # Load images dataframe
    df = enoe_utils.load_df(csv_path, place='SHOP')
    # Remove "bad" images (annotated as zero)
    df = df[ df['level']!=0 ]

    prev_row = None
    prev_img = None
    flow_df_rows = list()

    # Iterrate all rows
    for i, (index, row) in enumerate(df.iterrows()):
        # Read image given in the current row
        img = cv2.imread(os.path.join(enoe_dir,row['path']), cv2.IMREAD_GRAYSCALE)
        # Check if this is not the first image
        if prev_img is not None:
            # Check if the last image was taken less than 20 minutes ago
            time_diff_mins = (row['datetime'] - prev_row['datetime']).seconds//60
            if time_diff_mins < max_time_diff:
                # Compute optical flow, convert it into an image, and store it
                flow = compute_flow_pair(prev_img, img)
                flow_img = flow2img(flow)
                dir_name_u = os.path.dirname(os.path.join(flows_dir, 'u', row['path']))
                dir_name_v = os.path.dirname(os.path.join(flows_dir, 'v', row['path']))
                os.makedirs(dir_name_u, exist_ok=True)
                os.makedirs(dir_name_v, exist_ok=True)
                path_u = os.path.join('u', row['path'])
                path_v = os.path.join('v', row['path'])
                cv2.imwrite(os.path.join(flows_dir, path_u), flow_img[...,0])
                cv2.imwrite(os.path.join(flows_dir, path_v), flow_img[...,1])
                # Generate row for the new dataframe
                # New level = maximum level between the two frames
                new_lvl = max(row['level'], prev_row['level'])
                flow_df_rows.append([row['datetime'],
                                    row['place'],
                                    prev_row['path'],
                                    row['path'],
                                    path_u,
                                    path_v,
                                    prev_row['level'],
                                    row['level']])
        # Update prev variables
        prev_row = row
        prev_img = img
        # Show "alive" message every 500 iterations
        if not i%500:
            print(i, row)
    # Generate dataframe
    df_new = pd.DataFrame(flow_df_rows, columns=['datetime',
                                                'place',
                                                'path_prev',
                                                'path_next',
                                                'path_u',
                                                'path_v',
                                                'level_prev',
                                                'level_next'])
    df_new.to_csv(csv_flow_path)
