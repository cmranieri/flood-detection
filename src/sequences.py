from tensorflow.keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pandas as pd
import math
import os
import enoe_utils


class BaseEnoeSequence(Sequence):
    def __init__( self,
                  df, 
                  base_dir='/enoe', 
                  img_size=224,
                  batch_size=32, 
                  mode='train', 
                  flow=False,
                  samples_class_train=None,
                  max_samples_class_valid=None,
                  seed=1 ):
        np.random.seed(seed)
        self.base_dir = base_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.df = df
        if flow:
            self.df['level'] = self.df[['level_prev', 'level_next']].max(axis=1)
        if self.mode=='train' and samples_class_train is not None:
            self.df = enoe_utils.get_balanced_df( self.df, 
                                                  samples_class_train, 
                                                  seed=self.seed  )
        elif self.mode=='valid' and max_samples_class_valid is not None:
            self.df = enoe_utils.downsample_to_max( self.df,
                                                    max_samples_class_valid,
                                                    seed=self.seed )
        self.indices = np.arange( len(self.df) )
        if self.mode=='train':
            np.random.shuffle( self.indices )
        return

    def __len__( self ):
        return math.ceil( len(self.df)/self.batch_size )

    def on_epoch_end(self):
        if self.mode=='train':
            np.random.shuffle( self.indices )
        return


class SingleRGBSequence(BaseEnoeSequence):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        ids = self.indices[ index*self.batch_size :
                           (index+1)*self.batch_size ] 
        df_batch = self.df.iloc[ ids ]
        fnames = df_batch[ 'path' ].tolist()
        paths = [ os.path.join(self.base_dir,fname)
                      for fname in fnames ]
        images = np.array([ resize( imread(path), 
                                (self.img_size,self.img_size) )
                            for path in paths ])
        labels = np.array( df_batch[ 'level' ].tolist() )
        labels = to_categorical( labels-1, num_classes=4 )
        return images, labels


class SingleFlowSequence(BaseEnoeSequence):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        ids = self.indices[ index*self.batch_size :
                           (index+1)*self.batch_size ] 
        df_batch = self.df.iloc[ ids ]

        fnames_u = df_batch[ 'path_u' ].tolist()
        fnames_v = df_batch[ 'path_v' ].tolist()
        paths_u = [ os.path.join(self.base_dir,fname)
                    for fname in fnames_u ]
        paths_v = [ os.path.join(self.base_dir,fname)
                    for fname in fnames_v ]
        images_u = [ resize( imread(path, as_gray=True),
                         (self.img_size,self.img_size) )
                     for path in paths_u ]
        images_v = [ resize( imread(path, as_gray=True),
                         (self.img_size,self.img_size) )
                     for path in paths_v ]
        pairs = [ np.stack( [img_u, img_v], axis=-1 )
                  for img_u, img_v in zip(images_u, images_v) ]
        pairs = np.array(pairs)
        
        labels = np.array( df_batch[ 'level' ].tolist() )
        labels = to_categorical( labels-1, num_classes=4 )
        return pairs, labels
