# -*- coding: utf-8 -*-
"""
Code to train an RNN on the supervised audiovisual correspondence (SAVC) task 
with audiovisual CNN feature embeddings.
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

congruent_label = tf.Variable([0, 1], tf.uint8)
incongruent_label = tf.Variable([1, 0], tf.uint8)

config_path = 'train.yaml'

with open(config_path) as f:
    config = yaml.safe_load(f)


def build_audio_tfrecord_path(row):
    """ Build full audio tfrecord paths"""
    pos_path = tfrecords_loc1 + row['audio_tfrecord']
    pos_path = Path(pos_path)
    
    if pos_path.exists():
        return pos_path.as_posix()
    elif Path(tfrecords_loc2 + row['audio_tfrecord']).exists():
        return tfrecords_loc2 + row['audio_tfrecord']
    else:
        return None
    

def build_visual_tfrecord_path(row):
    """ Build full visual tfrecord paths"""
    
    pos_path = tfrecords_loc1 + row['visual_tfrecord']
    pos_path = Path(pos_path)
    
    if pos_path.exists():
        return pos_path.as_posix()
    elif Path(tfrecords_loc2 + row['visual_tfrecord']).exists():
        return tfrecords_loc2 + row['visual_tfrecord']
    else:
        return None


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


def euclidean_distance(vectors):
    """ Calculates Euclidean distance between 2 vectors.
    """
    vector1, vector2 = vectors
    sum_squared = tf.reduce_sum(
        tf.square(vector1 - vector2),
        axis=1,
        keepdims=True
    )
    return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))


def L2_bottleneck(
    input_layer: tf.keras.layers,
    timesteps: int,
    modality_name: str
):
    """ Add unimodal layers, prior to Euclidean distance calculation, to 
        graph.
    """
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(),
        name=f'{modality_name}_flatten'
    )(input_layer)
    
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128),
        name=f'{modality_name}_Dense_1'
    )(y)
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128),
        name=f'{modality_name}_Dense_2'
    )(y)
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
        name=f'{modality_name}_L2norm'
    )(y)
    
    return y


def add_congruent_label(audio, visual):
    """ Adds an incongruent label to each element in the data.Dataset"""
    return (audio, visual), congruent_label


def add_incongruent_label(audio, visual):
    """ Adds an incongruent label to each element in the data.Dataset"""
    return (audio, visual), incongruent_label


def get_audio_alone(audio, visual, int_lab):
    """ returns visual alone"""
    return audio


def get_visual_alone(audio, visual, int_lab):
    """ returns visual alone"""
    return visual



# %% Get data from configs
tfrecords_loc1 = config['tfrecords_loc1']
tfrecords_loc2 = config['tfrecords_loc2']

training_batch_size = config['training']['training_batch_size']
validation_batch_size = config['training']['validation_batch_size']
test_batch_size = config['training']['test_batch_size']
learning_rate = config['training']['learning_rate']
num_epochs = config['training']['num_epochs']

timesteps = config['data']['timesteps']

if config['data']['VGG_style']['enabled']:
    audio_size = config['data']['VGG_style']['audio_size']
    visual_size = config['data']['VGG_style']['visual_size']
elif config['data']['mobilenet_style']['enabled']:
    audio_size = config['data']['mobilenet_style']['audio_size']
    visual_size = config['data']['mobilenet_style']['visual_size']

model_path = config['model_path']


# %% Create model directory
Path(model_path).mkdir(exist_ok=False)


# %% Copy config to model directory
shutil.copyfile(config_path, model_path + 'train.yaml')


# %% read in CSVs
train_cong_df = pd.read_csv(config['congruent_train_csv'])
train_incong_df = pd.read_csv(config['incongruent_train_csv'])

val_cong_df = pd.read_csv(config['congruent_val_csv'])
val_incong_df = pd.read_csv(config['incongruent_val_csv'])


# %% Shuffle filepaths before reading from disk
train_cong_df = train_cong_df.sample(frac=1.0)
train_incong_df = train_incong_df.sample(frac=1.0)

val_cong_df = val_cong_df.sample(frac=1.0)
val_incong_df = val_incong_df.sample(frac=1.0)


# %% Prepare whole tfrecord paths for train set
train_cong_df['audio_tfrecord_path'] = train_cong_df.apply(
    build_audio_tfrecord_path,
    axis=1
)
train_cong_df['visual_tfrecord_path'] = train_cong_df.apply(
    build_visual_tfrecord_path,
    axis=1
)

train_incong_df['audio_tfrecord_path'] = train_incong_df.apply(
    build_audio_tfrecord_path,
    axis=1
)
train_incong_df['visual_tfrecord_path'] = train_incong_df.apply(
    build_visual_tfrecord_path,
    axis=1
)


# %% Prepare whole tfrecord paths for validation set
val_cong_df['audio_tfrecord_path'] = val_cong_df.apply(
    build_audio_tfrecord_path,
    axis=1
)
val_cong_df['visual_tfrecord_path'] = val_cong_df.apply(
    build_visual_tfrecord_path,
    axis=1
)

val_incong_df['audio_tfrecord_path'] = val_incong_df.apply(
    build_audio_tfrecord_path,
    axis=1
)
val_incong_df['visual_tfrecord_path'] = val_incong_df.apply(
    build_visual_tfrecord_path,
    axis=1
)


# %% Remove NaN values
train_cong_df.dropna(
    subset=['audio_tfrecord_path', 'visual_tfrecord_path'],
    how='any',
    inplace=True
)
train_incong_df.dropna(
    subset=['audio_tfrecord_path', 'visual_tfrecord_path'],
    how='any',
    inplace=True
)

val_cong_df.dropna(
    subset=['audio_tfrecord_path', 'visual_tfrecord_path'],
    how='any',
    inplace=True
)
val_incong_df.dropna(
    subset=['audio_tfrecord_path', 'visual_tfrecord_path'],
    how='any',
    inplace=True
)


# %% Read tfrecords into tf.data.Datasets and create congruent train set
train_cong_audio_ds = tf.data.TFRecordDataset(
    train_cong_df['audio_tfrecord_path'],
    compression_type='GZIP'
)
train_cong_visual_ds = tf.data.TFRecordDataset(
    train_cong_df['visual_tfrecord_path'],
    compression_type='GZIP'
)

train_cong_audio_ds = train_cong_audio_ds.map(parse_function) # parse tfrecord
train_cong_visual_ds = train_cong_visual_ds.map(parse_function) # parse tfrecord

train_cong_audio_ds = train_cong_audio_ds.map(get_audio_alone) # get audio data alone
train_cong_visual_ds = train_cong_visual_ds.map(get_visual_alone) # get visual data alone

train_cong_ds = tf.data.Dataset.zip(train_cong_audio_ds, train_cong_visual_ds)

train_cong_ds = train_cong_ds.map(add_congruent_label) # add label


# %% Read tfrecords into tf.data.Datasets and create INcongruent train set
train_incong_audio_ds = tf.data.TFRecordDataset(
    train_incong_df['audio_tfrecord_path'],
    compression_type='GZIP'
)
train_incong_visual_ds = tf.data.TFRecordDataset(
    train_incong_df['visual_tfrecord_path'],
    compression_type='GZIP'
)

train_incong_audio_ds = train_incong_audio_ds.map(parse_function) # parse tfrecord
train_incong_visual_ds = train_incong_visual_ds.map(parse_function) # parse tfrecord

train_incong_audio_ds = train_incong_audio_ds.map(get_audio_alone) # get audio data alone
train_incong_visual_ds = train_incong_visual_ds.map(get_visual_alone) # get visual data alone

train_incong_ds = tf.data.Dataset.zip(
    train_incong_audio_ds,
    train_incong_visual_ds
)

train_incong_ds = train_incong_ds.map(add_incongruent_label) # add label


# %% Create train set by sampling equally from cong and incong sets
train_ds = tf.data.Dataset.from_tensor_slices(
    [train_cong_ds, train_incong_ds]
).interleave(lambda x: x)


# %% Read tfrecords into tf.data.Datasets and create congruent validation set
val_cong_audio_ds = tf.data.TFRecordDataset(
    val_cong_df['audio_tfrecord_path'],
    compression_type='GZIP'
)
val_cong_visual_ds = tf.data.TFRecordDataset(
    val_cong_df['visual_tfrecord_path'],
    compression_type='GZIP'
)

val_cong_audio_ds = val_cong_audio_ds.map(parse_function) # parse tfrecord
val_cong_visual_ds = val_cong_visual_ds.map(parse_function) # parse tfrecord

val_cong_audio_ds = val_cong_audio_ds.map(get_audio_alone) # get audio data alone
val_cong_visual_ds = val_cong_visual_ds.map(get_visual_alone) # get visual data alone

val_cong_ds = tf.data.Dataset.zip(val_cong_audio_ds, val_cong_visual_ds)

val_cong_ds = val_cong_ds.map(add_congruent_label) # add label


# %% Read tfrecords into tf.data.Datasets and create INcongruent validation set
val_incong_audio_ds = tf.data.TFRecordDataset(
    val_incong_df['audio_tfrecord_path'],
    compression_type='GZIP'
)
val_incong_visual_ds = tf.data.TFRecordDataset(
    val_incong_df['visual_tfrecord_path'],
    compression_type='GZIP'
)

val_incong_audio_ds = val_incong_audio_ds.map(parse_function) # parse tfrecord
val_incong_visual_ds = val_incong_visual_ds.map(parse_function) # parse tfrecord

val_incong_audio_ds = val_incong_audio_ds.map(get_audio_alone) # get audio data alone
val_incong_visual_ds = val_incong_visual_ds.map(get_visual_alone) # get visual data alone

val_incong_ds = tf.data.Dataset.zip(
    val_incong_audio_ds,
    val_incong_visual_ds
)

val_incong_ds = val_incong_ds.map(add_incongruent_label) # add label


# %% Create train set by sampling equally from cong and incong sets
val_ds = tf.data.Dataset.from_tensor_slices(
    [val_cong_ds, val_incong_ds]
).interleave(lambda x: x)


# %% Shuffle the data and take data for epoch
train_ds = train_ds.shuffle(buffer_size=config['training']['buffer_size'])

train_ds = train_ds.take(12_960)
val_ds = val_ds.take(12_960)


# %% Batch data
train_ds = train_ds.padded_batch(
    training_batch_size,
    padded_shapes=(
        ((timesteps, audio_size), (timesteps, visual_size)),
        (2)
    )
)

val_ds = val_ds.padded_batch(
    validation_batch_size,
    padded_shapes=(
        ((timesteps, audio_size), (timesteps, visual_size)),
        (2)
    )
)


# %% Build model graph
audio_input = tf.keras.Input(
    shape=(timesteps, audio_size),
    name='audio_input'
) # audio input layer

visual_input = tf.keras.Input(
    shape=(timesteps, visual_size),
    name='visual_input' #name='visul_input'
) # visual input layer


model_inputs = [audio_input, visual_input]

if config['audio']:
    y = audio_input
    
if config['visual']:
    y = visual_input

if config['audio'] & config['visual']:
    
    audio_bottleneck = L2_bottleneck(audio_input, timesteps, 'audio')
    visual_bottleneck = L2_bottleneck(visual_input, timesteps, 'visual')
    
    y = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Lambda(euclidean_distance),
        name='Euclidean_distance'
    )([audio_bottleneck, visual_bottleneck])
    
    
rnn_layer = eval(config['rnn']['function'])
rnn = rnn_layer(
    **{k:v for k,v in config['rnn'].items() if k!='function'}
)(y)

final_dense = tf.keras.layers.Dense(2)(rnn)
outputs = tf.keras.layers.Activation(
    'softmax',
    dtype='float32',
    name='predictions'
)(final_dense) # get prob dist.


model = tf.keras.Model(
    inputs=model_inputs,
    outputs=[outputs]
)

optimizer = tf.keras.optimizers.Adam(learning_rate, weight_decay=10**-5)

metrics = [
    tf.keras.metrics.binary_accuracy
]

loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.run_eagerly=False
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
    run_eagerly=False
) # Compile model

model.summary()


# %% For repeatability
model.save_weights(model_path + 'random_seed.h5', save_format='h5') # rndm seed


# %%
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path + (
        'weights.{epoch:02d}-{val_loss:.2f}-{val_binary_accuracy:.2f}.ckpt'
    ),
    save_weights_only=True,
    monitor='val_binary_accuracy',
    mode='max',
    save_freq = 'epoch',
    save_best_only=True
)

callbacks = [model_checkpoint_callback]

history = model.fit(
    train_ds,
    epochs=num_epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=1 # data/epoch
)

model.save_weights(model_path + 'model', save_format='h5')