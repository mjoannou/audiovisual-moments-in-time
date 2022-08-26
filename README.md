# audiovisual-moments-in-time

The raw ratings are recorded in raw_video_ratings.csv. Each row corresponds to a video rating. As each video 
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

| Field          | Description                                     |
| -------------- |:-----------------------------------------------:|
| filename       | "MIT class subdirectory/ video name"            |
| r1             | number of '1' ratings given                     |
| r2             | number of '2' ratings given                     |
| r3             | number of '3' ratings given                     |
| AVMIT_label    | as displayed to participants in annotation task |
| MIT_label      | original dataset label                          |
| video_location | training or validation directories of MIT       |


The held-out test set details are provided in test_set.csv. These video were collected as described in 
*link to paper*, and are suitable for DNN vs human experiments in the audiovisual domain. Each row contains:

| Field          | Description                                     |
| -------------- |:-----------------------------------------------:|
| filename       | "MIT class subdirectory/ video name"            |
| AVMIT_label    | as displayed to participants in annotation task |
| MIT_label      | original dataset label                          |
| video_location | training or validation directories of MIT       |
| new_filename   | "AVMIT label subdirectory/ new video name"      |

