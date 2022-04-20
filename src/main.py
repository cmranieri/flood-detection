import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import callbacks
from sklearn import metrics
import argparse
import numpy as np
import pandas as pd
import os, shutil
import yaml
import enoe_utils, ml_utils
import sequences


def data_augmentation(config_augments):
    augments_dict = {
    'rotation':    layers.RandomRotation(factor=0.15),
    'translation': layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    'zoom':        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    'contrast':    layers.RandomContrast(factor=0.1)
    }
    img_augmentation = Sequential(
        [augments_dict[aug] for aug in config_augments],
        name="img_augmentation",)
    return img_augmentation


def build_model( config ):
    img_size = config['model']['img_size']
    input_ch = config['model']['input_channels']
    inputs = layers.Input( shape=(img_size, img_size, input_ch) )
    # Include layers for data augmentation, if required
    if config['train']['use_augments']:
        x = data_augmentation(config['train']['augmentations'])(inputs)
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
        optimizer = Adam(learning_rate = config['train']['lr'])
    elif config['train']['optimizer']=='sgd':
        optimizer = SGD(learning_rate = config['train']['lr'],
                        momentum = config['train']['sgd_momentum'])
    model.compile( optimizer = optimizer, 
                   loss      = config['train']['loss'],
                   metrics   = config['eval']['metrics'] )
    return model


def scheduler(epoch, lr):
    if epoch<10:
        return lr
    return lr * tf.math.exp(-0.1)


def train_valid_sequences(df_train, df_valid, config):
    if config['model']['sequence'] == 'SingleRGB':
        EnoeSequence = sequences.SingleRGBSequence
    elif config['model']['sequence'] == 'SingleFlow':
        EnoeSequence = sequences.SingleFlowSequence
    elif config['model']['sequence'] == 'SingleGrayFlow':
        EnoeSequence = sequences.SingleGrayFlowSequence
    elif config['model']['sequence'] == 'StackFlow':
        EnoeSequence = sequences.StackFlowSequence
    train_seq = EnoeSequence( 
        df                  = df_train,
        enoe_dir            = config['paths']['enoe_dir'],
        flow_dir            = config['paths']['flow_dir'],
        num_classes         = config['model']['num_classes'],
        img_size            = config['model']['img_size'],
        batch_size          = config['train']['batch_size'],
        k                   = config['model']['stack_k'],
        max_horizon_mins    = config['model']['max_horizon_mins'],
        mode                = 'train',
        seed                = config['experiment']['seed'] )
    valid_seq = EnoeSequence( 
        df                      = df_valid,
        enoe_dir                = config['paths']['enoe_dir'],
        flow_dir                = config['paths']['flow_dir'],
        num_classes             = config['model']['num_classes'],
        img_size                = config['model']['img_size'],
        batch_size              = config['eval']['batch_size'],
        k                       = config['model']['stack_k'],
        max_horizon_mins        = config['model']['max_horizon_mins'],
        mode                    = 'valid',
        seed                    = config['experiment']['seed'] )
    test_seq = EnoeSequence( 
        df               = df_valid,
        enoe_dir         = config['paths']['enoe_dir'],
        flow_dir         = config['paths']['flow_dir'],
        num_classes      = config['model']['num_classes'],
        img_size         = config['model']['img_size'],
        k                = config['model']['stack_k'],
        max_horizon_mins = config['model']['max_horizon_mins'],
        mode             = 'valid',
        batch_size       = config['eval']['batch_size'] )
    train_seq.fix_imbalance(
        samples_class_train = config['train']['samples_class_train'])
    valid_seq.fix_imbalance(
        max_samples_class_valid = config['train']['max_samples_class_valid'])
    return train_seq, valid_seq, test_seq
 

def eval_model(model, test_seq, model_dir, config):
    Y_pred = model.predict( test_seq,
                            verbose=1,
                            workers=config['eval']['workers'] )
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.array( test_seq.levels ) - 1
    balanced_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    cf = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report( y_true, y_pred,
                        target_names=config['model']['target_names'] )
    df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
    df.to_csv(os.path.join(model_dir,'preds.csv'))
    with open(os.path.join(model_dir,'results_summary.txt'),'w') as f:
        f.write(f'Balanced Accuracy: {balanced_acc}')
        f.write('\n\n\nConfusion Matrix\n\n')
        f.write(str(cf))
        f.write('\n\n\nClassification Report\n\n')
        f.write(str(report))
    print(f'Balanced Accuracy: {balanced_acc}')
    print('Confusion Matrix')
    print(cf)
    print('Classification Report')
    print(report)


def main(args):
    eval_only = args.eval_only
    config_path = args.config_path
    split = args.split

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_dir = os.path.join( config['paths']['models_dir'],
                              config['model_name'],
                              'split_'+str(split) )
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    log_dir = os.path.join(model_dir, 'logs')

    # Create dirs for checkpointing and logging
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load dataset and split
    df = enoe_utils.load_df( config['paths']['csv_path'],
                             place = 'SHOP',
                             flow  = config['model']['flow'],
                             use_diffs = config['model']['use_diffs'] )
    df_train, df_valid = enoe_utils.split_dataframe( df,
                                  split=split )

    # Define train and validation sequences
    train_seq, valid_seq, test_seq = train_valid_sequences(df_train, df_valid, config)

    # Build model or load from checkpoint
    initial_epoch = ml_utils.get_ckpt_epoch(checkpoint_dir)
    if initial_epoch:
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
        callbacks_list = [ callbacks.ModelCheckpoint(filepath=ckpt_path),
                           callbacks.TensorBoard(log_dir=log_dir),
                           callbacks.LearningRateScheduler(scheduler), ]
        # Train model
        hist = model.fit( train_seq,
                          validation_data=valid_seq,
                          epochs=config['train']['epochs'],
                          callbacks=callbacks_list,
                          initial_epoch=initial_epoch,
                          workers=config['train']['workers'] )
    # Keep only most recent checkpoints
    ml_utils.clear_old_ckpt(checkpoint_dir,
                            keep=config['train']['keep_ckpts'])
    # Evaluate model
    eval_model(model, test_seq, model_dir, config)
    return


if __name__ == '__main__':
    description = 'Flood detection based on computer vision.'
    default_path = '../configs/enoe_config.yaml'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--eval_only', type=bool, default=False,
                        help='If true, skip training, and\
                              evaluate the most recent checkpoint.')
    parser.add_argument('--config_path', type=str, default=default_path,
                        help='Path to the config file.')
    parser.add_argument('--split', type=int, default=1,
                        help='Train/validation split (1, 2, 3 or 4).')
    args = parser.parse_args()
    main(args)

