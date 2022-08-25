# audiovisual-moments-in-time

The held-out test set details are provided in test_set.csv. These video were collected as described in 
<paper link>, and are suitable for DNN vs human experiments in the audiovisual domain. Each row contains:

| Field          | Description                                     |
| -------------- |:-----------------------------------------------:|
| filename       | "MIT class subdirectory/ video name"            |
| AVMIT_label    | as displayed to participants in annotation task |
| MIT_label      | original dataset label                          |
| video_location | training or validation directories of MIT       |
| new_filename   | "AVMIT label subdirectory/ new video name"      |