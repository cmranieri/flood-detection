
import pandas as pd
import numpy as np


def load_df( csv_path, place=None, flow=False ):
    # Read dataframe from csv
    df = pd.read_csv( csv_path, parse_dates=['datetime'], index_col=0 )
    df.loc[ df['place'].isna(), 'place' ] = 'unknown'
    # Convert levels to numeric and discard invalid images
    if not flow:
        df[ 'level' ] = pd.to_numeric(df['level'] )
        df.loc[ df['level']==5, 'level' ] = np.nan
        df = df[ ~df['level'].isna() ]
    # There are no invalid images in the optical flow csv
    else:
        df[ 'level_prev' ] = pd.to_numeric(df['level'] )
        df[ 'level_next' ] = pd.to_numeric(df['level'] )
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
