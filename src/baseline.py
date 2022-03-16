import pandas as pd
import numpy as np
import math
import os
import shutil
import re
import yaml
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
import enoe_utils, ml_utils


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


def build_model( config ):
    img_size = config['model']['img_size']
    inputs = layers.Input( shape=(img_size, img_size, 3) )
    # Include layers for data augmentation, if required
    if config['train']['use_augments']:
        x = data_augmentation()(inputs)
        input_tensor = x
    else:
        input_tensor = inputs
    # Load base model
    if config['model']['base_model']=='EfficientNetB0':
        model = EfficientNetB0( include_top = False,
                                input_tensor = input_tensor, 
                                weights = config['train']['weights'] )
    elif config['model']['base_model']=='ResNet50':
        model = ResNet50( include_top = False,
                          input_tensor = input_tensor, 
                          weights = config['train']['weights'] )
    # Freeze the pretrained weights, if required
    if config['train']['finetune']:
        model.trainable=False
    # Rebuild top
    x = layers.GlobalAveragePooling2D()(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['model']['top_dropout'])(x)
    outputs = layers.Dense( config['model']['num_classes'],
                            activation='softmax' )(x)
    # Compile
    model = tf.keras.Model(inputs, outputs)
    if config['train']['optimizer']=='adam':
        optimizer = Adam(learning_rate=config['train']['lr'])
    model.compile( optimizer = optimizer, 
                   loss      = config['train']['loss'],
                   metrics   = config['eval']['metrics'] )
    return model


def eval_model(model, valid_seq, model_dir, config):
    Y_pred = model.predict(valid_seq)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.array( valid_seq.df['level'] ) - 1
    cf = confusion_matrix(y_true, y_pred)
    report = classification_report( y_true, y_pred,
                        target_names=config['model']['target_names'] )
    np.savetxt(os.path.join(model_dir,'cf.csv'), cf)
    with open(os.path.join(model_dir,'results_summary.txt'),'w') as f:
        f.write('Confusion Matrix\n\n')
        f.write(str(cf))
        f.write('\n\n\nClassification Report\n\n')
        f.write(str(report))
    print('Confusion Matrix')
    print(cf)
    print('Classification Report')
    print(report)




if __name__ == '__main__':
    eval_only = False
    config_path = '../configs/enoe_config.yaml'

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_dir = os.path.join( config['paths']['models_dir'],
                              config['model_name'] )
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    log_dir = os.path.join(model_dir, 'logs')

    # Create dirs for checkpointing and logging
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load dataset and split
    df = enoe_utils.load_df(config['paths']['csv_path'], place='SHOP')
    df_train, df_val = enoe_utils.split_dataframe( df,
                                split=config['experiment']['split'] )

    # Define train and validation generators
    train_seq = EnoeSequence( df = df_train,
                              base_dir=config['paths']['data_dir'],
                              img_size=config['model']['img_size'],
                              batch_size=config['train']['batch_size'],
                              mode='train',
                              seed=config['experiment']['seed'] )
    valid_seq = EnoeSequence( df = df_val,
                              base_dir=config['paths']['data_dir'],
                              img_size=config['model']['img_size'],
                              batch_size=config['eval']['batch_size'],
                              mode='valid',
                              seed=config['experiment']['seed'] )
    
    # Build model or load from checkpoint
    initial_epoch = ml_utils.get_ckpt_epoch(checkpoint_dir)
    if initial_epoch:
        ml_utils.clear_old_ckpt(checkpoint_dir)
        model = load_model(os.path.join(checkpoint_dir,
                                f'model.{initial_epoch:02d}.h5'))
    else:
        model = build_model( config )

    # Train model
    if not eval_only:
        # Copy yaml configuration file
        shutil.copyfile(config_path, 
                os.path.join(model_dir, 'backup_config.yaml'))
        # Setup callbacks
        ckpt_path = os.path.join(checkpoint_dir,'model.{epoch:02d}.h5')
        callbacks_list = [ ModelCheckpoint(filepath=ckpt_path),
                           TensorBoard(log_dir=log_dir), ]
        # Train model
        hist = model.fit( train_seq,
                          validation_data=valid_seq,
                          epochs=config['train']['epochs'],
                          callbacks=callbacks_list,
                          initial_epoch=initial_epoch,
                          workers=config['train']['workers'] )
        ml_utils.clear_old_ckpt(checkpoint_dir)

    # Evaluate model
    eval_model(model, valid_seq, model_dir, config)
