#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to detect whether each video, in AVMIT or MIT-16, has an audio stream.
"""

import pandas as pd
from pydub import AudioSegment

# %% filepaths

MIT_dir = '/Moments_in_Time_Raw/' # Path to original Moments in Time dataset

AVMIT_train_df = pd.read_csv(
    'AVMIT_train.csv'
)
AVMIT_val_df = pd.read_csv(
    'AVMIT_validation.csv'
)

MIT16_train_df = pd.read_csv(
    'MIT16_train.csv'
)
MIT16_val_df = pd.read_csv(
    'MIT16_validation.csv'
)

# %% concat AVMIT train and val together, and MIT16 train and val sets

AVMIT_df = pd.concat([AVMIT_train_df, AVMIT_val_df])
MIT16_df = pd.concat([MIT16_train_df, MIT16_val_df])

# %% Build full video paths for AVMIT

def build_video_path(row):
    if row['video_location'] == 'MIT_training':
        split = 'training'
    elif row['video_location'] == 'MIT_validation':
        split = 'validation'
    return MIT_dir + split + '/' + row['filename']

AVMIT_df['full_video_path'] = AVMIT_df.apply(build_video_path, axis=1)

# %% Build full video paths for MIT-16

def build_video_path(row):
    split = 'training'
    return MIT_dir + split + '/' + row['filename']

MIT16_df['full_video_path'] = MIT16_df.apply(build_video_path, axis=1)

# %% For each video obtain a boolean value for presence of audio in AVMIT

for index, row in AVMIT_df.iterrows():
    vid_path = row['full_video_path']
    
    mask = AVMIT_df['filename'] == row['filename']
    
    try:
        seg = AudioSegment.from_file(vid_path, 'mp4')
        AVMIT_df.loc[mask, 'audio?'] = True
        print(f'contains audio in {row["full_video_path"]}')
    except FileNotFoundError as e:
        print(f'no audio in {row["full_video_path"]}')
        AVMIT_df.loc[mask, 'audio?'] = False
    
# %% Again for MIT16

num_vids = len(MIT16_df.index)

for index, row in MIT16_df.iterrows():
    print(f'{index}/{num_vids}')
    vid_path = row['full_video_path']
    
    mask = MIT16_df['filename'] == row['filename']
    
    try:
        seg = AudioSegment.from_file(vid_path, 'mp4')
        MIT16_df.loc[mask, 'audio?'] = True
        print(f'contains audio in {row["full_video_path"]}')
    except:
        print(f'no audio in {row["full_video_path"]}')
        MIT16_df.loc[mask, 'audio?'] = False