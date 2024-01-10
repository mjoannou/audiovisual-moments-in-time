#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to measure the audiovisual similarity of the audio and visual stream of 
each video in a dataset using multimodal versatile network (MMV;
Alayrac JB, et al. Self-Supervised Multimodal Versatile Networks. 
In: Advances in Neural Information Processing Systems (NeurIPS); 2020.)
"""
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import resampy
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm

num_secs = 3.2

# Audio params
desired_sample_rate = 48_000
desired_audio_samples = desired_sample_rate * num_secs
desired_sample_width = 2 # corresponds to 2 bytes (16 bit audio)

# Visual params
frame_shape = (200, 200, 3) # frame res

ds_dir = Path('AVE_PATH/') # script will glob all mp4s in this dir


def get_centre_frames(frame_ls, sample_sz):
    """ Takes an iterable (of frames/audio samples) and returns sample_sz 
        samples from the centre.
    """
    mid_idx = int(len(frame_ls)/2)
    half_sample_sz = int(sample_sz/2)
    
    lower_idx = mid_idx - half_sample_sz
    upper_idx = mid_idx + half_sample_sz
    
    return frame_ls[lower_idx:upper_idx]


def preprocess_audio(filename):
    """ Preprocesses audio stream from a video file ready for S3D"""
    audio_seg = AudioSegment.from_file(filename)

    audio_seg = audio_seg.set_channels(1) # obtain mono audio
    audio_seg = audio_seg.set_sample_width(
        desired_sample_width) # set bit depth to 16

    audio_array = np.asarray(
        audio_seg.get_array_of_samples()
    ) # from audioseg to array
    audio_array = audio_array.astype(dtype="float32") # Cast to float32
    
    audio_array = audio_array / 32_768.0  # Convert to [-1.0, +1.0]
    
    sample_rate = audio_seg.frame_rate # get sample rate
    if sample_rate != desired_sample_rate: # if needs to be resampled
        audio_array = resampy.resample(
            audio_array,
            sample_rate,
            desired_sample_rate
        ) # resample
    
    if len(audio_array) > desired_audio_samples:
        audio_array = get_centre_frames(audio_array, desired_audio_samples)
    
    audio_array = np.expand_dims(audio_array, axis=0) # batch dim
    
    return audio_array
    

def get_visual_frames(filename, frame_rate_ms=100):
    """ Preprocesses video frames ready for S3D"""
    cap = cv2.VideoCapture()
    
    if not cap.open(filename):
        print(f'{sys.stderr}: Cannot open {filename}')
        return
    
    last_ts = -99_999  # timestamp of last retrieved frame
    end_of_video = False # flag to end
    
    frames = []
    while True:
        # throw away frames until it is time to collect another frame
        while (cap.get(cv2.CAP_PROP_POS_MSEC) < frame_rate_ms + last_ts):
            if not cap.read()[0]:
                end_of_video = True # activate flag
                break
        if end_of_video: # detect flag
            break
        
        last_ts = cap.get(cv2.CAP_PROP_POS_MSEC) # Timestamp of this frame
        has_frames, frame = cap.read()
        if not has_frames: # If the video is over
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
        frame = cv2.resize(frame, dsize=frame_shape[:2]) # Resize the frame
        frame = frame.astype(float) / 255 # [0, 1]
        
        frames.append(frame)
    
    if len(frames) > 32:
        frames = get_centre_frames(frames, 32)
        
    frames = np.stack(frames, axis=0)
    
    return frames


# %% Function to calculate similarity
def compute_similarity_scores(filename_ls):
    """ Iterates over video filenames and computes audiovisual correspondence
        for each video.
    """    
    f_sim_dict = {} # populated by for loop
    
    for vid_path in tqdm(filename_ls):
        
        try:
            input_audio = preprocess_audio(vid_path) # prepare audio
            
            audio_output = model.signatures['audio'](tf.constant(
                tf.cast(input_audio, dtype=tf.float32))
            ) # audio infer
            
            audio_embedding = audio_output['va'] # audio embs
            
        except:
            print(f'sim not calculated for {vid_path}')
            continue
            
        try:
            input_frames = get_visual_frames(vid_path) # prepare visual       
            
            vision_output = model.signatures['video'](tf.constant(
                tf.cast(input_frames, dtype=tf.float32))
            ) # visual infer
            
            video_embedding = vision_output['va'] # visual embs
            
        except:
            print(f'sim not calculated for {vid_path}')
            continue
            
        # Compute all the pairwise similarity scores between video and audio.
        sim_matrix = tf.matmul(
            audio_embedding,
            video_embedding,
            transpose_b=True
        )
        
        sim_score = sim_matrix.numpy().mean()
        
        vid = '/'.join(Path(vid_path).parts[-2:])# train/file.mp4
        f_sim_dict[vid] = sim_score # add sim_score to dict
        
        print(f'Similarity score: {sim_score}')
        
    return pd.DataFrame(
        f_sim_dict.items(),
        columns=['filename', 'similarity score']
    )


path_ls = [f.as_posix() for f in ds_dir.glob('*.mp4')]

model = hub.load(
    'https://www.kaggle.com/models/deepmind/mmv/frameworks/'
    'TensorFlow1/variations/s3d/versions/1'
) # Load S3D

sims = compute_similarity_scores(path_ls)