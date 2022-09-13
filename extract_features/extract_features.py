# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:17:37 2022

@author: Anonymous

Code to extract audiovisual features with 
pretrained convolutional neural networks and 
create a tfrecord dataset.
"""

# %% imports
import sys
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import resampy
import cv2
import argparse
import os


# %% preprocess audio data
def preprocess_audio(filename):
    try: # try and read audio stream
        audio_seg = AudioSegment.from_file(filename)
    except IndexError: # If there is no audio stream
        # create digital silence
        audio_array = np.zeros(
            (num_secs*desired_sample_rate),
            dtype="float32")
    else: # if audio stream opened successfully
        audio_seg = audio_seg.set_channels(1) # obtain mono audio
        audio_seg = audio_seg.set_sample_width(
            desired_sample_width) # set bit depth to 16
        audio_array = np.asarray(
            audio_seg.get_array_of_samples(),
            dtype="float32") # Cast from audioseg to array
        audio_array = audio_array / 32768.0  # Convert to [-1.0, +1.0]
        
        sample_rate = audio_seg.frame_rate # get sample rate
        if sample_rate != desired_sample_rate: # if needs to be resampled
            audio_array = resampy.resample(
                audio_array,
                sample_rate,
                desired_sample_rate) # resample
    return audio_array


# %%  preprocess visual frames
def yield_visual_frames(filename, frame_rate_ms=480):
    cap = cv2.VideoCapture() # create video capture object
    if not cap.open(filename): # open video
        print(f'{sys.stderr}: Cannot open {filename}') # inform the user
        return
    
    last_ts = -99999  # timestamp of last retrieved frame
    end_of_video = False # flag to end
    
    while True: # until the loop is explicitly broken
        # throw away frames until it is time to collect another frame
        while cap.get(cv2.CAP_PROP_POS_MSEC) < frame_rate_ms + last_ts:
            # read frame (throwaway frame) if cannot be opened, video is over
            if not cap.read()[0]:
                end_of_video = True # activate flag
                break
        if end_of_video: # detect flag
            break
        
        last_ts = cap.get(cv2.CAP_PROP_POS_MSEC) # Timestamp of this frame
        has_frames, frame = cap.read() # Read the current frame
        if not has_frames: # If the video is over
            break # Break the loop
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
        frame = cv2.resize(frame, dsize=frame_shape[:2]) # Resize the frame
        frame = preprocess_image(frame)
        yield frame


# %% functions to prepare data for protocol buffer
def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_feature_list(value_list):
    return tf.train.FeatureList(
        feature=[bytes_feature(v.tostring()) for v in value_list])


# %% loop through videos, obtain audiovisual embeddings, write to file
def main():
    cntr = 1
    num_videos = len(videos.index)
    for filename, avmit_label, filepath in zip(videos['filename'],
                                               videos['AVMIT_label'],
                                               videos['full_filepath']):

        print(f'{cntr}/ {num_videos}\n{filepath}') # inform user of progress
                
        # check video file exists
        if not Path(filepath).is_file(): # if file does not exist
            # is the filepath > 260 characters (MAX_PATH for DOS)
            if len(filepath) > 260:                
                filepath = (Path(filepath).parent / shortened_filepath[
                    Path(filepath).name]).as_posix() # use filename alias
            else:
                with open(vids_not_included, 'a', newline='') as f:
                    f.write(filepath) # record filepath of unopened file
                    print(f'{filepath} does not exist\n')
                continue # Move to the next video

        # build new tfrecord filename
        tfrecord_filepath = (f'{tfrecord_dir}/'
                             f'{avmit_label}/'
                             f'{Path(filepath).stem}'
                             f'.tfrecord')
        
        # check tfrecord does not already exist
        if Path(tfrecord_filepath).exists(): # if tfrecord exists
            print("{} exists".format(tfrecord_filepath)) # inform user
            cntr += 1
            continue # next video
        
        # create subdir for this label if doesn't exist
        tfrecord_subdir = Path(tfrecord_filepath).parents[0] # label subdir
        if not tfrecord_subdir.exists(): # if subdir does not exist
            tfrecord_subdir.mkdir(parents=False) # create subdir        
        
        # obtain audio embeddings
        audio_stream = preprocess_audio(Path(filepath))

        if audio_model_name == 'VGGish':
            audio_embeddings = audio_model(audio_stream) # shape (ts, 128)
        elif audio_model_name == 'YamNet':
            _, audio_embeddings, _ = audio_model(audio_stream) # (ts, 1024)
            
        audio_embeddings = audio_embeddings.numpy() # EagerTensor > np.ndarray
        
        # obtain visual embeddings
        visual_embeddings = np.asarray([
            visual_model(np.expand_dims(i, axis=0)) 
            for i in yield_visual_frames(filepath, frame_interval)
            ]) # np.ndarray of visual model frame embeddings, size 512 or 1024
        visual_embeddings = np.squeeze(
            visual_embeddings, axis=1) # (timesteps,1,512) > (timesteps,512)
        
        print(audio_embeddings.shape)
        print(visual_embeddings.shape)
    
        # prepare the non-sequential data (tf.train.feature) for proto buf
        context_dict = {'filename': bytes_feature(filename.encode('utf-8')),
                        'label': bytes_feature(avmit_label.encode('utf-8')),
                        'audio_timesteps': int_feature(
                            audio_embeddings.shape[0]),
                        'visual_timesteps': int_feature(
                            visual_embeddings.shape[0])
                        }
        
        # prepare the sequential data (tf.train.featurelist) for proto buf
        feature_lists_dict = {
            'audio': bytes_feature_list(audio_embeddings),
            'visual_object': bytes_feature_list(visual_embeddings)
        }
        
        # prepare protocol buffer
        seq_example = tf.train.SequenceExample(
            context = tf.train.Features(feature=context_dict),
            feature_lists = tf.train.FeatureLists(
                feature_list=feature_lists_dict)
            )
    
        # prepare options for compression of tfrecords
        options = tf.io.TFRecordOptions( # Ensure tfrecords are zipped
        compression_type='GZIP', flush_mode=None, input_buffer_size=None,
        output_buffer_size=None, window_bits=None, compression_level=None,
        compression_method=None, mem_level=None, compression_strategy=None
        )         
        
        # write protocol buffer to tfrecords file
        writer = tf.io.TFRecordWriter(tfrecord_filepath, options=options)
        writer.write(seq_example.SerializeToString()) # serialise & write
        
        # last tfrecord does not close and is always corrupt, 
        # create this throwaway tfrecord
        writer = tf.io.TFRecordWriter(
            tfrecord_dir+'/delete_this.tfrecord',
            options=options)
        writer.write(seq_example.SerializeToString()) # Write throwaway file
    
        cntr += 1


if __name__ == '__main__':
    
    # %% argparse to receive arguments
    parser = argparse.ArgumentParser(description=(
        'This program extracts audio and visual embeddings from the videos '
        'listed in video_ratings.csv and writes them to a tfrecord dataset.'))
    parser.add_argument('-repo_dir',
                        required = True,
                        help=('Please provide the path to the repo '
                        'e.g. C:/username/audiovisual-moments-in-time'))
    parser.add_argument('-MIT_dir',
                        required = True,
                        help=('Please provide the path to the MIT dataset '
                        'e.g. C:/username/Moments_in_Time_Raw'))
    parser.add_argument('-tfrecord_dir',
                        required = True,
                        help=('Please provide the output directory path for '
                        'the tfrecord dataset e.g. C:/username/tfrecord_ds'))
    parser.add_argument('-audio_model',
                        required = True,
                        choices=['VGGish', 'YamNet'],
                        help=('Please provide the audio CNN to be used '
                        'to extract the audio embeddings.'))
    parser.add_argument('-visual_model',
                        required = True,
                        choices=['VGG16', 'EfficientNetB0'],
                        help=('Please provide the visual CNN to be used '
                        'to extract the visual embeddings.'))
    
    args = parser.parse_args()
    
    repo_dir = args.repo_dir
    MIT_dir = args.MIT_dir
    tfrecord_dir = args.tfrecord_dir
    audio_model_name = args.audio_model
    visual_model_name = args.visual_model


    # %% long filepaths - use FAT32-style 8-character filename aliases
    shortened_filepath = {
        ('vb-scout-master-and-boys-drilling-ice-fishing-hole-frozen-lake-'
         'winter-boy-scout-outing-and-camp-where-they-went-ice-fishing-on-'
         'fish-lake-during-winter-water-frozen-two-feet-thick-nice-spring-day'
         '-but-cold-snow-on-surface-and-mountains-tndrgkp_9.mp4'):
            'VB-SCO~1.mp4',
        ('vb-paraglider-trike-trying-to-takeoff-on-a-taxiway-at-a-small-'
         'airport-the-parachute-would-not-inflate-and-the-pilot-aborts-his-'
         'flight-bright-yellow-and-blue-chute-extreme-sports-and-recreation'
         '-don-despain-of-rekindle-photo-q5oehry_6.mp4'):
            'VB-PAR~2.mp4',
        ('vb-des-plaines-illinois-usa-2011-07-23-2-record-amount-of-rainfall'
         '-fell-overnight-in-chicago-and-suburbs-des-plaines-river-level-rose'
         '-very-quickly-and-eventually-overflowed-flooding-many-streets-and'
         '-houses-le4fxmt_1.mp4'):
            'VB-DES~2.mp4',
        ('vb-des-plaines-illinois-usa-2011-07-23-3-record-amount-of-rainfall'
         '-fell-overnight-in-chicago-and-suburbs-des-plaines-river-level-rose'
         '-very-quickly-and-eventually-overflowed-flooding-many-streets-and'
         '-houses-hogh80q_9.mp4'):
            'VB-DES~1.mp4',
        ('vb-des-plaines-illinois-usa-2011-07-23-4-record-amount-of-rainfall-'
         'fell-overnight-in-chicago-and-suburbs-des-plaines-river-level-rose'
         '-very-quickly-and-eventually-overflowed-flooding-many-streets-and'
         '-houses-urqbdpd_1.mp4'):
            'VB-DES~4.mp4',
        ('vb-des-plaines-illinois-usa-2011-07-23-record-amount-of-rainfall-'
         'fell-overnight-in-chicago-and-suburbs-des-plaines-river-level-rose'
         '-very-quickly-and-eventually-overflowed-flooding-many-streets-and'
         '-houses-d8mzfe-_2.mp4'):
            'VB-DES~3.mp4',
        ('vb-neighborhood-under-water-3-this-is-one-of-many-flooded-street-'
         'due-to-the-overflowing-des-plaines-river-after-heavy-rainfalls-at-'
         'that-time-flood-stage-in-this-area-was-96ft-major-flooding-stage-'
         'is-90ft-508t0k5_3.mp4'):
            'VB-NEI~3.mp4',
        ('vb-neighborhood-under-water-4-this-is-one-of-many-flooded-street-'
         'due-to-the-overflowing-des-plaines-river-after-heavy-rainfalls-at'
         '-that-time-flood-stage-in-this-area-was-96ft-major-flooding-stage'
         '-is-90ft-xwlh4nw_4.mp4'):
            'VB-NEI~2.mp4'
        }

    # %% other filepaths 
    # record of videos not prepared into tfrecords
    vids_not_included = tfrecord_dir + '/videos_not_included.txt'
    video_csv = f'{repo_dir}/video_ratings.csv' # location of video filepaths
    

    # %% preprocessing parameters
    num_secs = 3 # for silent audio generation
    desired_sample_rate = 16000 # sample rate for VGGish/YamNet
    desired_sample_width = 2 # corresponds to 2 bytes (16 bit audio)
    
    frame_interval = 480 # time (ms) between visual frames
    
    frame_shape = (224, 224, 3) # frame res
    
    # %% obtain video list 
    videos = pd.read_csv(video_csv)
    
    # %% prepare audio model
    if audio_model_name == 'VGGish':
        audio_model = hub.load('https://tfhub.dev/google/vggish/1')
    elif audio_model_name == 'YamNet':
        audio_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # %% prepare visual model
    if visual_model_name == 'VGG16':
        preprocess_image = tf.keras.applications.vgg16.preprocess_input
        visual_model = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            pooling='avg',
            input_shape=frame_shape)
    elif visual_model_name == 'EfficientNetB0':
        preprocess_image = lambda x: x # pass-through function
        visual_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            pooling='avg',
            input_shape=frame_shape)
    
    # %% obtain full video filepaths
    videos['full_filepath'] = videos.apply(
        lambda row: (f'{MIT_dir}'
                     f'{row["video_location"]}/{row["filename"]}'), axis=1)

    # %% call main() loop to extract features and write to file
    main()