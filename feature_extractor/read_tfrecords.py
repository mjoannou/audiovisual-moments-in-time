# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 09:10:23 2022

@author: Anonymous

Example code to quickly load data from tfrecords into data.Dataset.
"""

# %% imports
from pathlib import Path
import tensorflow as tf

# %% prepare test dataset using stimuli directory
tfrecord_dir = 'C:/Users/username/tfrecord_directory' # location of tfrecords


# %% function to parse tfrecord dataset
def parse_function(serialised_example):
    # mapping function to parse data from protocol buffer
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
    
    return filename, audio, visual, label


# %% glob all tfrecord filenames into a list
filename_list = [
    vid.as_posix() for subdir in Path(tfrecord_dir).glob('*') 
    for vid in subdir.glob('*') if subdir.is_dir()]

# %% build data.Dataset
dataset_1 = tf.data.TFRecordDataset(filename_list, compression_type='GZIP')
dataset_1 = dataset_1.map(parse_function)