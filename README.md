# "Audiovisual Moments in Time: A Large-Scale Annotated Dataset of Audiovisual Actions"

This repository contains code corresponding to https://arxiv.org/abs/2308.09685. Feature embeddings can be found at https://zenodo.org/record/8253350.

Feature extractor code (used to produce the embeddings) and example code to show how to read the embeddings can be found alongside instructions in feature_extractor/.

csv files containing annotations and the test set list can be found in this repository (raw_video_ratings.csv, video_ratings.csv and test_set.csv) as well as in our zenodo dataset (https://zenodo.org/record/8253350)

The core idea of the work is to provide a dataset of feature embeddings and audiovisual annotations to train deep neural networks at low computational cost and to be able to carry out controlled experiments in the 
audiovisual domain (sampling from modality-agnostic action recognition datasets would not give exclusively audiovisual events). We also provide filenames corresponding to annotations should researchers wish to access raw videos 
from the well-established Moments in Time dataset (MIT; Monfort et al., 2019) which was used in our annotation regime and used to obtain our feature embeddings. To obtain MIT, one needs to visit http://moments.csail.mit.edu/ and 
fill out a form before access to the dataset is sent via email.

We provide 3 independent participant ratings for 57,177 videos (and embeddings) and further identify a highly controlled audiovisual test set of 960 videos across 16 action classes, suitable for both DNN and human experimentation.

Whilst we provide the filenames of the videos alongside their rating data, we additionally provide ready-made audiovisual embeddings of the videos There are 2 sets of audiovisual embeddings; those 
obtained using VGGish (Hershey et al., 2017) and VGG16 (Simonyan and Zisserman, 2015) and a second set obtained using YamNet (Plakal and Ellis, 2020) and EfficientNetB0 (Tan and Le, 2019). We provide the 
code to obtain these embeddings in feature_extractor/extract_features.py. This code can be editted to obtain embeddings using different tensorflow models if you wish (I am also open to requests for 
the provision of particular embeddings providing that the pretrained models are available via TensorFlow). The filename:feature-embedding correspondences are found in the csv files described below.

# Data

The raw ratings are recorded in raw_video_ratings.csv. Each row corresponds to a video rating by a single participant. As each video 
was rated 3 times, each video has 3 separate rows, distributed throughout the file. Each row contains:

| Field          | Description                                     |
| -------------- |:-----------------------------------------------:|
| filename       | "MIT class subdirectory/ video name"            |
| AVMIT_label    | as displayed to participants in annotation task |
| MIT_label      | original dataset label                          |
| video_location | training or validation directories of MIT       |
| rating         | the rating given by the trained participant     |


The accumulated ratings are recorded in video_ratings.csv. Each row corresponds to a video (containing 
all corresponding ratings). Each video was rated 3 times (r1+r2+r3=3). Videos rated less than 3 times were 
removed. Each row contains:

| Field            | Description                                                               |
| ---------------- |:-------------------------------------------------------------------------:|
| filename         | "MIT class subdirectory/ video name"                                      |
| r1               | number of '1' ratings given                                               |
| r2               | number of '2' ratings given                                               |
| r3               | number of '3' ratings given                                               |
| AVMIT_label      | as displayed to participants in annotation task                           |
| MIT_label        | original dataset label                                                    |
| video_location   | training or validation directories of MIT                                 |
| tfrecord_filename| subdirectory and filename of corresponding audiovisual feature embeddings |

The held-out test set details are provided in test_set.csv. These video were collected as described in 
https://arxiv.org/abs/2308.09685, and are suitable for DNN vs human experiments in the audiovisual domain. 
Each row contains:

| Field            | Description                                                               |
| ---------------- |:-------------------------------------------------------------------------:|
| filename         | "MIT class subdirectory/ video name"                                      |
| AVMIT_label      | as displayed to participants in annotation task                           |
| MIT_label        | original dataset label                                                    |
| video_location   | training or validation directories of MIT                                 |
| new_filename     | "AVMIT label subdirectory/ new video name"                                |
| tfrecord_filename| subdirectory and filename of corresponding audiovisual feature embeddings |


# References

Tan, Mingxing and Quoc V. Le (2019). “EfficientNet: Rethinking model scaling for convolutional neural networks”. In: 36th International Conference on Machine Learning, ICML 2019. Vol. 2019-June, pp. 10691–10700. ISBN: 9781510886988.

Hershey, Shawn et al. (2017). “CNN architectures for large-scale audio classification”. In: IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings (ICASSP). ISBN: 9781509041176. DOI: 10.1109/ICASSP.2017.7952132.

Monfort, Mathew et al. (2019). “Moments in Time Dataset: One Million Videos for Event Understanding”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 42.2, pp. 502–508. DOI: 10.1109/TPAMI.2019.2901464.

Plakal, Manoj and Dan Ellis (2020). YAMNet. URL: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet.

Simonyan, Karen and Andrew Zisserman (Sept. 2015). “Very Deep Convolutional Networks for Large-Scale Image Recognition”. In: International Conference of Learning Representations (ICLR). URL: http://arxiv.org/abs/1409.1556.
