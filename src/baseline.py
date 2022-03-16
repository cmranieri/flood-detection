import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras import callbacks

from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import numpy as np
import math
import os
import enoe_utils
import re


class EnoeSequence( Sequence ):

    def __init__( self,
                  df, 
                  base_dir, 
                  img_size,
                  num_samples_train=2000,
                  max_samples_valid=1000,
                  batch_size=32, 
                  mode='train', 
                  seed=1 ):
        self.base_dir = base_dir
        self.img_size = img_size
        self.num_samples_train = num_samples_train
        self.max_samples_valid = max_samples_valid
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.df = df
        if self.mode=='train':
            self.df = self.get_balanced_df( num_samples_train )
        elif self.mode=='valid':
            self.df = self.downsample_to_max( max_samples_valid )
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


def data_augmentation():
    img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",)
    return img_augmentation


def build_model( img_size=224, num_classes=4, augmentations=False ):
    inputs = layers.Input( shape=(img_size, img_size, 3) )
    input_tensor = inputs
    if augmentations:
        x = data_augmentation()(inputs)
        input_tensor = x
    model = ResNet50( include_top=False, 
                      input_tensor=input_tensor, 
                      weights="imagenet" )
    # Freeze the pretrained weights
    #model.trainable = False
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile( optimizer=optimizer, 
                   loss="categorical_crossentropy",
                   metrics=["accuracy"] )
    return model


def get_initial_epoch( checkpoint_dir ):
    epochs_list = [0]
    for fname in os.listdir(checkpoint_dir):
        mtc = re.match( r'.*model\.(\d+)', fname )
        if not mtc:
            continue
        epochs_list.append(int( mtc.groups()[0]) )
    return max(epochs_list)


if __name__ == '__main__':
    img_size = 224
    seed = 1
    epochs = 40
    augmentations = True
    csv_path='../resources/flood_images_annot.csv'
    model_name = 'baseline_v0'
    checkpoint_dir = f'/models/checkpoints/{model_name}'
    log_dir = f'/models/logs/{model_name}'

    # Create dirs for checkpointing and logging
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load dataset and split
    df = enoe_utils.load_df(csv_path, place='SHOP')
    df_train, df_val = enoe_utils.split_dataframe( df )

    # Define train and validation generators
    train_seq = EnoeSequence( df = df_train,
                              base_dir='/enoe',
                              img_size=img_size,
                              batch_size=32,
                              seed=seed )
    valid_seq = EnoeSequence( df = df_val,
                              base_dir='/enoe',
                              img_size=img_size,
                              batch_size=32,
                              mode='valid',
                              seed=seed )
    
    # Build model or load from checkpoint
    initial_epoch = get_initial_epoch( checkpoint_dir )
    if initial_epoch:
        model = load_model( os.path.join(checkpoint_dir,
                                         f'model.{initial_epoch:02d}.h5') )
    else:
        initial_epoch = 0
        model = build_model( img_size=img_size,
                             num_classes=4,
                             augmentations=augmentations )

    # Setup callbacks
    checkpoint_path = os.path.join(checkpoint_dir,'model.{epoch:02d}.h5')
    callbacks_list = [ callbacks.ModelCheckpoint(filepath=checkpoint_path),
                       callbacks.TensorBoard(log_dir=log_dir), ]
    # Train model
    hist = model.fit( train_seq,
                      validation_data=valid_seq,
                      epochs=epochs,
                      callbacks=callbacks_list,
                      initial_epoch=initial_epoch,
                      workers=8 )
 
