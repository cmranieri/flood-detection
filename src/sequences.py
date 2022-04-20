from tensorflow.keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pandas as pd
import math
import os
from scipy import stats
import enoe_utils


class BaseEnoeSequence(Sequence):
    def __init__( self,
                  df, 
                  enoe_dir='/enoe', 
                  flow_dir='/flow', 
                  num_classes=4,
                  img_size=224,
                  batch_size=32, 
                  mode='train', 
                  seed=1 ):
        np.random.seed(seed)
        self.enoe_dir = enoe_dir
        self.flow_dir = flow_dir
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.df = df
        self.levels = self.df['level']
        self.setup_indices()
        return

    def __len__(self):
        return math.ceil( len(self.df)/self.batch_size )

    def on_epoch_end(self):
        if self.mode=='train':
            np.random.shuffle( self.indices )
        return

    def setup_indices(self):
        self.indices = np.arange( len(self.df) )
        if self.mode=='train':
            np.random.shuffle( self.indices )

    def fix_imbalance( self, 
                       samples_class_train=None, 
                       max_samples_class_valid=None ):
        if self.mode=='train' and samples_class_train is not None:
            self.df = enoe_utils.generate_balanced( self.df, 
                                                    samples_class_train, 
                                                    num_classes=self.num_classes,
                                                    seed=self.seed  )
        elif self.mode=='valid' and max_samples_class_valid is not None:
            self.df = enoe_utils.downsample_to_max( self.df,
                                                    max_samples_class_valid,
                                                    num_classes=self.num_classes,
                                                    seed=self.seed )
        self.setup_indices()
        return


class SingleRGBSequence(BaseEnoeSequence):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        ids = self.indices[ index*self.batch_size :
                           (index+1)*self.batch_size ] 
        df_batch = self.df.iloc[ ids ]
        fnames = df_batch[ 'path' ].tolist()
        paths = [ os.path.join(self.enoe_dir,fname)
                      for fname in fnames ]
        images = np.array([ resize( imread(path), 
                                (self.img_size,self.img_size) )
                            for path in paths ])
        labels = np.array( df_batch[ 'level' ].tolist() )
        labels = to_categorical( labels-1, num_classes=self.num_classes )
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
        paths_u = [ os.path.join(self.flow_dir,fname)
                    for fname in fnames_u ]
        paths_v = [ os.path.join(self.flow_dir,fname)
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
        labels = to_categorical( labels-1, num_classes=self.num_classes )
        return pairs, labels


class SingleGrayFlowSequence(BaseEnoeSequence):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        ids = self.indices[ index*self.batch_size :
                           (index+1)*self.batch_size ] 
        df_batch = self.df.iloc[ ids ]
        fnames   = df_batch[ 'path_next' ].tolist()
        fnames_u = df_batch[ 'path_u' ].tolist()
        fnames_v = df_batch[ 'path_v' ].tolist()
        paths_g = [ os.path.join(self.enoe_dir,fname)
                    for fname in fnames ]
        paths_u = [ os.path.join(self.flow_dir,fname)
                    for fname in fnames_u ]
        paths_v = [ os.path.join(self.flow_dir,fname)
                    for fname in fnames_v ]
        images_g = [ resize( imread(path, as_gray=True),
                         (self.img_size,self.img_size) )
                     for path in paths_g ]
        images_u = [ resize( imread(path, as_gray=True),
                         (self.img_size,self.img_size) )
                     for path in paths_u ]
        images_v = [ resize( imread(path, as_gray=True),
                         (self.img_size,self.img_size) )
                     for path in paths_v ]
        inputs = [ np.stack( [img_g, img_u, img_v], axis=-1 )
                    for img_g, img_u, img_v in zip(images_g, images_u, images_v) ]
        inputs = np.array(inputs)
        labels = np.array( df_batch[ 'level' ].tolist() )
        labels = to_categorical( labels-1, num_classes=self.num_classes )
        return inputs, labels

class StackFlowSequence(BaseEnoeSequence):
    def __init__( self, 
                  k=3,
                  max_horizon_mins=120,
                  **kwargs ):
        super().__init__(**kwargs)
        self.k = k
        self.max_horizon_mins = max_horizon_mins
        self.df = self.generate_stacks( k, max_horizon_mins )
        self.setup_indices()
        
    def __getitem__(self, index):
        ids = self.indices[ index*self.batch_size :
                           (index+1)*self.batch_size ] 
        df_batch = self.df.iloc[ ids ]
        images = list()
        for i in range(self.k):
            fnames_u = df_batch[f'path_u_{i}'].to_list()
            fnames_v = df_batch[f'path_v_{i}'].to_list()
            paths_u = [ os.path.join(self.flow_dir,fname)
                        for fname in fnames_u ]
            paths_v = [ os.path.join(self.flow_dir,fname)
                        for fname in fnames_v ]
            images.append( [ resize( imread(path, as_gray=True),
                                 (self.img_size,self.img_size) )
                             for path in paths_u ] )
            images.append( [ resize( imread(path, as_gray=True),
                                (self.img_size,self.img_size) )
                             for path in paths_v ] )
        stacks = np.transpose(images, [1,2,3,0])
        labels = np.array( df_batch[ 'level' ].tolist() )
        labels = to_categorical( labels-1, num_classes=self.num_classes )
        return stacks, labels

    def generate_stacks(self, k, max_horizon_mins):
        datetimes = list(); places = list()
        paths_u = list(); paths_v = list(); paths_g = list()
        levels  = list()
        df = self.df.sort_values( by=['datetime'], ascending=[True] )
        for i in range(k-1, len(df)):
            horizon_mins = (df.iloc[i]['datetime']-df.iloc[i-k]['datetime']).seconds//60
            if horizon_mins < max_horizon_mins:
                datetimes.append(df.iloc[i]['datetime'])
                places.append(df.iloc[i]['place'])
                paths_u.append([df.iloc[i-k+1:i+1]['path_u'].to_list()])
                paths_v.append([df.iloc[i-k+1:i+1]['path_v'].to_list()])
                paths_g.append([df.iloc[i-k+1:i+1]['path_next'].to_list()])
                lvls_i = df.iloc[i-k+1:i+1]['level'].to_list()
                levels.append( max(lvls_i))
        paths_g = np.array(paths_g).squeeze()
        paths_u = np.array(paths_u).squeeze()
        paths_v = np.array(paths_v).squeeze()
        levels  = np.array(levels).squeeze()
        # Generate new dataframe
        new_df  = pd.DataFrame()
        new_df['datetime'] = datetimes
        new_df['place'] = places
        for i in range(k):
            new_df[f'path_u_{i}'] = list(paths_u[:,i])
            new_df[f'path_v_{i}'] = list(paths_v[:,i])
            new_df[f'path_g_{i}'] = list(paths_g[:,i])
            new_df['level'] = list(levels)
        print(new_df['level'].to_list())
        return new_df


if __name__=='__main__':
    df = enoe_utils.load_df( '../resources/flood_flow_annot.csv',
                             place = 'SHOP',
                             flow  = True,
                             use_diffs=True )
    df_train, df_valid = enoe_utils.split_dataframe( df, split=2 )
    seq = StackFlowSequence( df=df_train  )
    seq.__getitem__(16)

