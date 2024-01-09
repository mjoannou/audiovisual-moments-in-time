# -*- coding: utf-8 -*-
"""
Code to train an RNN on audiovisual action recognition using the feature 
embeddings of the AVMIT dataset.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import tensorflow as tf
from pathlib import Path
import pandas as pd
import yaml
from functools import partial
import shutil

tf.random.set_seed(1234)

config_path = 'train.yaml'


def build_tfrecord_path(row):
    """ Build full tfrecord paths"""
    return tfrecords_loc + row['tfrecord_filename']


def parse_function(serialised_example):
    """ mapping function to parse data from protocol buffer (tfrecords)"""
    context, features = tf.io.parse_single_sequence_example(
        serialised_example,
        context_features={
            'filename': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'audio_timesteps': tf.io.FixedLenFeature([], tf.int64),
            'visual_timesteps': tf.io.FixedLenFeature([], tf.int64)
            },
        sequence_features={
            'audio': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
            'visual_object': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
            }
    )
    
    filename = context['filename']
    label = context['label']
    audio_timesteps = context['audio_timesteps']
    visual_timesteps = context['visual_timesteps']
    
    audio = tf.io.decode_raw(features['audio'], tf.float32)
    audio = tf.reshape(audio, [audio_timesteps, -1])
    
    visual = tf.io.decode_raw(features['visual_object'], tf.float32)
    visual = tf.reshape(visual, [visual_timesteps, -1])
    
    return audio, visual, label


def label_str2int(label_mapping, audio, visual, str_lab):
    """ converts tf.string labels to tf.int label from hashtable"""
    int_lab = label_mapping[str_lab]
    return audio, visual, int_lab


def get_one_hots(audio, visual, int_lab):
    """ map integer label to one-hot encoding"""
    one_hot_lab = tf.one_hot(int_lab, num_classes, dtype=tf.uint8)
    return (audio, visual), one_hot_lab


def bottleneck(
    input_layer: tf.keras.layers,
    timesteps: int,
    modality_name: str
):
    """ add bottleneck stream to graph"""
    y = tf.keras.layers.Reshape(
        (timesteps, 1, 1, -1),
        name=f'{modality_name}_reshape'
    )(input_layer)
    
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(bottleneck_size, 1),
        name=f'{modality_name}_bottleneck'
    )(y)
    
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.BatchNormalization(),
        name=f'{modality_name}_bn'
    )(y)
    
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Activation(bottleneck_activation, dtype='float32'),
        name=f'{modality_name}_activation'
    )(y)
    
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GlobalAveragePooling2D(),
        name=f'{modality_name}_pooling'
    )(y)
    
    return y


# %% Read train config
with open(config_path) as f:
    config = yaml.safe_load(f)


# %% Get data from configs
tfrecords_loc = config['tfrecords_loc']
label_mapping_path = config['label_to_int']
num_classes = config['data']['num_classes'] # AVMIT classes
buffer_size = config['training']['buffer_size']
training_batch_size = config['training']['training_batch_size']
validation_batch_size = config['training']['validation_batch_size']
test_batch_size = config['training']['test_batch_size']
learning_rate = config['training']['learning_rate']
early_stopping = config['training']['early_stopping']
num_epochs = config['training']['num_epochs']

timesteps = config['data']['timesteps']

if config['data']['VGG_style']['enabled']:
    audio_size = config['data']['VGG_style']['audio_size']
    visual_size = config['data']['VGG_style']['visual_size']
elif config['data']['mobilenet_style']['enabled']:
    audio_size = config['data']['mobilenet_style']['audio_size']
    visual_size = config['data']['mobilenet_style']['visual_size']
    
label2int = config['label_to_int']
bottleneck_size = config['bottleneck_size']
bottleneck_activation = config['bottleneck_activation']

model_path = config['model_path']
log_dir = config['log_dir']

train_csv = config['train_csv']
val_csv = config['val_csv']

# %% Create model directory
Path(model_path).mkdir(exist_ok=False)

# %% Copy config to model directory
shutil.copyfile(config_path, model_path + 'train.yaml')

# %% Load csv files
train_df = pd.read_csv(train_csv) # 410 per class
val_df = pd.read_csv(val_csv) # 46 per class

# %% build data.Datasets
train_df['tfrecord_path'] = train_df.apply(build_tfrecord_path, axis=1)
val_df['tfrecord_path'] = val_df.apply(build_tfrecord_path, axis=1)

train_ds = tf.data.TFRecordDataset(
    train_df['tfrecord_path'],
    compression_type='GZIP'
)

val_ds = tf.data.TFRecordDataset(
    val_df['tfrecord_path'],
    compression_type='GZIP'
)

train_ds = train_ds.map(parse_function)
val_ds = val_ds.map(parse_function)

# %% Convert string labels to integers

# turn dictionary to tensorflow hashtable
keys = tf.constant(list(label2int.keys()), dtype=tf.string)
values = tf.constant(list(label2int.values()), dtype=tf.int64)
initializer = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)

htable = tf.lookup.StaticHashTable(
    initializer=initializer,
    default_value = tf.constant(0, dtype=tf.int64)
) # for label lookup

label_str2int = partial(label_str2int, htable)

train_ds = train_ds.map(label_str2int)
val_ds = val_ds.map(label_str2int)

# %% Get the one hots from integers
train_ds = train_ds.map(
    get_one_hots,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).cache()

val_ds = val_ds.map(
    get_one_hots,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).cache()

# %% Shuffle the data
train_ds = train_ds.shuffle(buffer_size=buffer_size)
val_ds = val_ds.shuffle(buffer_size=buffer_size)

# %% Batch data
train_ds = train_ds.padded_batch(
    training_batch_size,
    padded_shapes=(
        ((timesteps, audio_size), (timesteps, visual_size)),
        num_classes
    )
)

val_ds = val_ds.padded_batch(
    validation_batch_size,
    padded_shapes=(
        ((timesteps, audio_size), (timesteps, visual_size)),
        num_classes
    )
)

# %% Build model graph
audio_input = tf.keras.Input(
    shape=(timesteps, audio_size),
    name='audio_input'
) # audio input layer

visual_input = tf.keras.Input(
    shape=(timesteps, visual_size),
    name='visul_input'
) # visual input layer


model_inputs = [audio_input, visual_input]

if config['audio']:
    y = audio_input
    
if config['visual']:
    y = visual_input

if config['audio'] & config['visual']:
    audio_bottleneck = bottleneck(audio_input, timesteps, 'audio')
    visual_bottleneck = bottleneck(visual_input, timesteps, 'visual')
    
    y = tf.keras.layers.concatenate(
        [audio_bottleneck, visual_bottleneck],
        axis=2
    )
    
rnn_layer = eval(config['rnn']['function'])
rnn = rnn_layer(
    **{k:v for k,v in config['rnn'].items() if k!='function'}
)(y)

final_dense = tf.keras.layers.Dense(num_classes)(rnn)
outputs = tf.keras.layers.Activation(
    'softmax',
    dtype='float32',
    name='predictions'
)(final_dense) # get prob dist.


model = tf.keras.Model(
    inputs=model_inputs,
    outputs=[outputs]
)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
metrics = [
    tf.keras.metrics.categorical_accuracy,
    tf.keras.metrics.TopKCategoricalAccuracy(
        k=5,
        name='top_5_categorical_accuracy',
        dtype=None
    )
]

model.run_eagerly=False
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=metrics,
    run_eagerly=False
) # Compile model

model.summary()


# %% For repeatability
model.save_weights(model_path + 'random_seed.h5', save_format='h5') # rndm seed

# %%
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=early_stopping,
    restore_best_weights=True,
    min_delta=0.001
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path + 'weights.{epoch:02d}-{val_loss:.2f}.ckpt',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_freq = 'epoch',
    save_best_only=True)

callbacks = [
    early_stopping_callback,
    model_checkpoint_callback
]

history = model.fit(
    train_ds,
    epochs=num_epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=2 # data/epoch
)

model.save_weights(model_path + 'model', save_format='h5')