from tensorflow.keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pandas as pd
import math
import os


class SingleRGBSequence( Sequence ):
    def __init__( self,
                  df, 
                  base_dir, 
                  img_size,
                  samples_class_train=None,
                  max_samples_class_valid=None,
                  batch_size=32, 
                  mode='train', 
                  seed=1 ):
        np.random.seed(seed)
        self.base_dir = base_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.df = df
        if self.mode=='train' and samples_class_train is not None:
            self.df = self.get_balanced_df( samples_class_train )
        elif self.mode=='valid' and max_samples_class_valid is not None:
            self.df = self.downsample_to_max( max_samples_class_valid )
        self.indices = np.arange( len(self.df) )
        return

    def downsample_to_max( self, max_samples ):
        # Provide list of dataframes for each level
        sample_dfs = list()
        for level in [ 1, 2, 3, 4 ]:
            df_level = self.df[ self.df['level']==level ]
            # Downsample overrepresented categories
            if len( df_level ) > max_samples:
                df_level = df_level.sample( n=max_samples,
                                            replace=False,
                                            random_state=self.seed )
            sample_dfs.append( df_level )
        new_df = pd.concat( sample_dfs )
        return new_df
    
    def get_balanced_df( self, num_samples ):
        # Provide list of balanced dataframes for each level
        sample_dfs = list()
        for level in [ 1, 2, 3, 4 ]:
            df_level = self.df[ self.df['level']==level ]
            # Upsample underrepresented categories
            if len( df_level ) < num_samples:
                aux_df = pd.DataFrame( np.repeat( df_level.values,
                                                  math.ceil(num_samples/len(df_level)), 
                                                  axis=0 ) )
                aux_df.columns = self.df.columns
                df_level = aux_df
            # Downsample overrepresented categories
            if len( df_level ) > num_samples:
                df_level = df_level.sample( n=num_samples,
                                            replace=False,
                                            random_state=self.seed )
            sample_dfs.append( df_level )
        balanced_df = pd.concat( sample_dfs )
        return balanced_df

    def __len__( self ):
        return math.ceil( len(self.df)/self.batch_size )

    def __getitem__(self, index):
        ids = self.indices[ index*self.batch_size :
                           (index+1)*self.batch_size ] 
        df_batch = self.df.iloc[ ids ]
        filenames = df_batch[ 'path' ].tolist()
        filenames = [ os.path.join(self.base_dir,fname)
                      for fname in filenames ]
        images = np.array([ resize(imread(fname), (self.img_size,self.img_size))
                            for fname in filenames ])
        labels = np.array( df_batch[ 'level' ].tolist() )
        labels = to_categorical( labels-1, num_classes=4 )
        return images, labels

    def on_epoch_end(self):
        if self.mode=='train':
            np.random.shuffle( self.indices )
        return

