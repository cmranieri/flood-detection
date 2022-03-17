from tensorflow.keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pandas as pd
import math
import os
import enoe_utils


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
            self.df = enoe_utils.get_balanced_df( df, 
                                                  samples_class_train, 
                                                  seed=seed  )
        elif self.mode=='valid' and max_samples_class_valid is not None:
            self.df = enoe_utils.downsample_to_max( df,
                                                    max_samples_class_valid,
                                                    seed=seed )
        self.indices = np.arange( len(self.df) )
        return

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

