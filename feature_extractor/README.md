# Feature Extractor

This directory contains the code (extract_features.py) to extract audiovisual CNN embeddings from the videos annotated 
in this work, listed in video_ratings.csv. We also provide read_tfrecords.py as an example of how to read in and parse 
the tfrecords so that users can get started quickly.

# Dependencies

The following versions or later

Python 3.7.9

- tensorflow 2.3.1
- tensorflow_hub 0.12.0
- pandas 1.1.3
- numpy 1.18.5
- pydub 0.24.1
- pathlib 1.0.1
- resampy 0.2.2
- opencv-python 4.4.0.44

# Usage

Run extract_features.py from the terminal or Anaconda prompt and provide required filepaths and choice of CNN model. 
`-repo_dir` is the location of this repo on your local 
machine, `-MIT_dir` is the location of the unzipped Moments in Time dataset, `-tfrecord_dir` is the 
desired location for the tfrecord set to be saved (you should create an empty folder first), 
`-audio_model` can be either `VGGish` or `YamNet` and `-visual_model` can be either `VGG16` or `EfficientNetB0`. 
Note that while we provide VGG-based embeddings or efficientnet/mobilenet-based embeddings, you can use any 
combination of the audio and visual models.

Example:
```
python extract_features.py -repo_dir C:/username/audiovisual-moments-in-time -MIT_dir C:/username/Moments_in_Time_Raw -tfrecord_dir C:/username/tfrecord_ds -audio_model VGGish -visual_model VGG16
```