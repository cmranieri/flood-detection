
import pandas as pd
import numpy as np

def load_df( csv_path, place=None ):
    # Read dataframe from csv
    df = pd.read_csv( csv_path, parse_dates=['datetime'], index_col=0 )
    df[ 'level' ] = pd.to_numeric(df['level'] )
    df.loc[ df['place'].isna(), 'place' ] = 'unknown'
    # Discard invalid images
    df.loc[ df['level']==5, 'level' ] = np.nan
    df = df[ ~df['level'].isna() ]
    # Filter camera location
    if place is not None:
        df = df[ df['place'].str.contains(place) ]
    # Sort images by datetime
    df = df.sort_values( by=['datetime'], ascending=[True] )
    return df


def split_dataframe( df ):
    df_valid = df[ (df['datetime'] > pd.to_datetime('2019-11-01')) &
                   (df['datetime'] < pd.to_datetime('2020-02-01')) ]
    df_train = pd.concat( [df,df_valid] ).drop_duplicates( keep=False )
    return df_train, df_valid
