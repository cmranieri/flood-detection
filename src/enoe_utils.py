import pandas as pd
import numpy as np
import math
from scipy import stats


def load_df( csv_path, place=None, flow=False, use_diffs=False ):
    # Read dataframe from csv
    df = pd.read_csv( csv_path, parse_dates=['datetime'], index_col=0 )
    df.loc[ df['place'].isna(), 'place' ] = 'unknown'
    # Convert levels to numeric and discard invalid images
    if not flow:
        df[ 'level' ] = pd.to_numeric(df['level'] )
        df.loc[ df['level']==5, 'level' ] = np.nan
        df.loc[ df['level']==0, 'level' ] = np.nan
        df = df[ ~df['level'].isna() ]
    # There are no invalid images in the optical flow csv
    else:
        df[ 'level_prev' ] = pd.to_numeric(df['level_prev'] )
        df[ 'level_next' ] = pd.to_numeric(df['level_next'] )
        if use_diffs:
            # Force diff values to be in {1,2,3}, meaning "down", "still" or "up"
            df['level'] = df['level_next'] - df['level_prev']
            df['level'] = df['level'].clip(lower=-1, upper=1) + 2
        else:
            df['level'] = df[['level_next']]
    # Filter camera location
    if place is not None:
        df = df[ df['place'].str.contains(place) ]
    # Sort images by datetime
    df = df.sort_values( by=['datetime'], ascending=[True] )
    return df


def split_dataframe( df, split=2 ):
    if not split in [1,2,3,4]:
        raise Exception('There are only 4 splits available (1,2,3,4)')
    if split==1:
        start='2018-11-01'; end='2019-03-01'
    elif split==2:    
        start='2019-11-01'; end='2020-03-01'
    elif split==3:    
        start='2020-11-01'; end='2021-03-01'
    elif split==4:    
        start='2021-11-01'; end='2022-03-01'
    df_valid = df[ (df['datetime'] > pd.to_datetime(start)) &
                   (df['datetime'] < pd.to_datetime(end)) ]
    df_train = pd.concat( [df,df_valid] ).drop_duplicates( keep=False )
    return df_train, df_valid

def downsample_to_max( df, max_samples, num_classes=4, seed=1 ):
    # Provide list of dataframes for each level
    sample_dfs = list()
    for level in np.arange(1,num_classes+1):
        df_level = df[ df['level']==level ]
        # Downsample overrepresented categories
        if len( df_level ) > max_samples:
            df_level = df_level.sample( n=max_samples,
                                        replace=False,
                                        random_state=seed )
        sample_dfs.append( df_level )
    new_df = pd.concat( sample_dfs )
    return new_df

def generate_balanced( df, num_samples, num_classes=4, seed=1 ):
    # Provide list of balanced dataframes for each level
    sample_dfs = list()
    for level in np.arange(1,num_classes+1):
        df_level = df[ df['level']==level ]
        # Upsample underrepresented categories
        if len( df_level ) < num_samples:
            aux_df = pd.DataFrame( np.repeat( df_level.values,
                                                math.ceil(num_samples/len(df_level)), 
                                                axis=0 ) )
            aux_df.columns = df.columns
            df_level = aux_df
        # Downsample overrepresented categories
        if len( df_level ) > num_samples:
            df_level = df_level.sample( n=num_samples,
                                        replace=False,
                                        random_state=seed )
        sample_dfs.append( df_level )
    balanced_df = pd.concat( sample_dfs )
    return balanced_df

def generate_stacks(df, k=3, max_horizon_mins=120):
    paths_u = list()
    paths_v = list()
    paths_g = list()
    levels  = list()
    df = df.sort_values( by=['datetime'], ascending=[True] )
    for i in range(k-1, len(df)):
        horizon_mins = (df.iloc[i]['datetime']-df.iloc[i-k]['datetime']).seconds//60
        if horizon_mins < max_horizon_mins:
            paths_u.append([df.iloc[i-k+1:i+1]['path_u'].to_list()])
            paths_v.append([df.iloc[i-k+1:i+1]['path_v'].to_list()])
            paths_g.append([df.iloc[i-k+1:i+1]['path_next'].to_list()])
            lvls_i = df.iloc[i-k+1:i+1]['level'].to_list()
            print(df.iloc[i-k+1:i+1])
            print(lvls_i, stats.mode(lvls_i).mode[0])
            levels.append(stats.mode(lvls_i).mode[0])
    paths_g = np.array(paths_g).squeeze()
    paths_u = np.array(paths_u).squeeze()
    paths_v = np.array(paths_v).squeeze()
    levels  = np.array(levels).squeeze()
    print(df['level'])
    print(levels)
    return paths_g, paths_u, paths_v, levels

