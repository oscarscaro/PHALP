#!/bin/bash

# for a shot change video, try --youtube_id "2VxpRal7wJE" 

python demo_online_multi.py \
--track_dataset      "test" \
--storage_folder     "testing_multi_disabled" \
--predict            "TPL" \
--distance_type      "EQ_010" \
--encode_type        "4c" \
--detect_shots       True \
--all_videos         True \
--track_history      7 \
--past_lookback      20 \
--max_age_track      4 \
--n_init             5 \
--low_th_c           0.8 \
--alpha              0.1 \
--hungarian_th       100 \
--render_type        "HUMAN_FULL_FAST" \
--render             True \
--res                256 \
--render_up_scale    2 \
--verbose            False \
--overwrite          True \
--use_gt             False \
--batch_id           -1 \
--detection_type     "mask" \
--start_frame        -1 \
--end_frame          100 \
--multi_view         True \
--num_views          2 \
--test_video_id      "Multi_view_test2"