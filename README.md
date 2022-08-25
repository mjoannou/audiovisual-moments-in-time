# audiovisual-moments-in-time

The ratings are recorded in participant_ratings.csv. Each row corresponds to a video rating. As each video was 
rated 3 times, each video has 3 separate rows, distributed throughout the file. Each row contains:

| Field          | Description                                     |
| -------------- |:-----------------------------------------------:|
| filename       | "MIT class subdirectory/ video name"            |
| AVMIT_label    | as displayed to participants in annotation task |
| MIT_label      | original dataset label                          |
| video_location | training or validation directories of MIT       |
| rating         | the rating given by the trained participant     |

The held-out test set details are provided in test_set.csv. These video were collected as described in 
*link to paper*, and are suitable for DNN vs human experiments in the audiovisual domain. Each row contains:

| Field          | Description                                     |
| -------------- |:-----------------------------------------------:|
| filename       | "MIT class subdirectory/ video name"            |
| AVMIT_label    | as displayed to participants in annotation task |
| MIT_label      | original dataset label                          |
| video_location | training or validation directories of MIT       |
| new_filename   | "AVMIT label subdirectory/ new video name"      |

