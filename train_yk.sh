# nohup bash odtrack_train.sh >>./log/train_odtrack256_ab_layer_12.out&
gpu="0"

# chmod +x train_yk.sh
# ./train_yk.sh

# echo '----------------------------- Train ODTrack_256_hial  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_hial --save_dir ./output --mode single --nproc_per_node 1 

# # echo '-----------------------------  Test ODTrack_256_hial -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_hial --dataset HiAL_test --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_hial --dataset HiAL_test --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_hial --dataset HiAL_test --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_hial --dataset HiAL_test --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_hial --dataset HiAL_test --runid 11 --threads 6 --num_gpus 1

echo '----------------------------- Train ODTrack_256_lasher  -----------------------------'
CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_lasher --save_dir ./output --mode single --nproc_per_node 1 

echo '-----------------------------  Test ODTrack_256_lasher -----------------------------'
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 15 --threads 6 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 14 --threads 6 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 13 --threads 6 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 12 --threads 6 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 11 --threads 6 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 10 --threads 4 --num_gpus 1

# echo '-----------------------------  Analysis baseline_256_lasher  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 15
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 14
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 13
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 12
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 11


#  echo '----------------------------- Train ODTrack_256 for RGBT234 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_lasher4rgbt234 --save_dir ./output --mode single --nproc_per_node 1 

# echo '-----------------------------  Test ODTrack_256 for RGBT234 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt234 --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt234 --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt234 --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt234 --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt234 --runid 11 --threads 6 --num_gpus 1


# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt210 --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt210 --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt210 --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt210 --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset rgbt210 --runid 11 --threads 6 --num_gpus 1


#  echo '----------------------------- Train ODTrack_256 for RGBT210 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_lasher4rgbt210 --save_dir ./output --mode single --nproc_per_node 1 

# echo '-----------------------------  Test ODTrack_256 for RGBT210 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt234 --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 15 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt234 --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 14 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt234 --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 13 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt234 --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 12 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt234 --runid 11 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 11 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 11 --threads 6 --num_gpus 1




# echo '----------------------------- Train ODTrack_256_VTUAV  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_vtuav --save_dir ./output --mode single --nproc_per_node 1 

# echo '-----------------------------  Test ODTrack_256_VTUAV -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset vtuav_test --runid 15 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset vtuav_test --runid 14 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset vtuav_test --runid 13 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset vtuav_test --runid 12 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset vtuav_test --runid 11 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset lasher_test --runid 10 --threads 4 --num_gpus 1

# echo '-----------------------------  Analysis ODTrack_256_VTUAV  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset_name vtuav_test --runid 15
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset_name vtuav_test --runid 14
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset_name vtuav_test --runid 13
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset_name vtuav_test --runid 12
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset_name vtuav_test --runid 11

# echo '-----------------------------  Test ODTrack_256_lasher for RGBT210 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt210 --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 14 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt210 --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 15 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt210 --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 13 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt210 --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 12 --threads 6 --num_gpus 1

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt210 --runid 11 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 11 --threads 6 --num_gpus 1

# echo '-----------------------------  Analysis baseline_256_lasher for RGBT210  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name rgbt210 --runid 14

# echo '-----------------------------  Test ODTrack_256_lasher for RGBT234 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 14 --threads 4 --num_gpus 1

# echo '-----------------------------  Analysis baseline_256_lasher for RGBT234  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name rgbt234 --runid 14
# ====================================================================================================================================================================================
# echo '384版本，需要更改 lib/models/odtrack/odtrack.py 102行！！！！'
# echo '----------------------------- Train ODTrack_384  -----------------------------'
# # CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_384_lasher --save_dir ./output --mode single --nproc_per_node 1 

# echo '-----------------------------  Test ODTrack_384  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset lasher_test --runid 15 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset lasher_test --runid 14 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset lasher_test --runid 13 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset lasher_test --runid 12 --threads 6 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset lasher_test --runid 11 --threads 6 --num_gpus 1
# echo '-----------------------------  Analysis ODTrack_384  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset_name lasher_test --runid 15

# ====================================================================================================================================================================================

# echo '----------------------------- Train ODTrack_256 on VTUAV -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_vtuav --save_dir ./output --mode multiple --nproc_per_node 2

# echo '-----------------------------  Test ODTrack_256  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset vtuav_test --runid 15 --threads 4 --num_gpus 2

# echo '-----------------------------  Analysis ODTrack_256  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset_name vtuav_test --runid 15

# ====================================================================================================================================================================================

# echo '384版本，需要更改 lib/models/odtrack/odtrack.py 102行！！！！'
# echo '----------------------------- Train ODTrack_384  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_384_vtuav --save_dir ./output --mode multiple --nproc_per_node 2

# echo '-----------------------------  Test ODTrack_384  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_vtuav --dataset vtuav_test --runid 15 --threads 4 --num_gpus 2

# echo '-----------------------------  Analysis ODTrack_384  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_384_vtuav --dataset_name vtuav_test --runid 15

# ====================================================================================================================================================================================

# echo '----------------------------- Train ODTrack_256 for RGBT234 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/train.py --script odtrack --config baseline_256_lasher4rgbt210 --save_dir ./output --mode multiple --nproc_per_node 2

# echo '-----------------------------  Test ODTrack_256 for RGBT234 -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt234 --runid 15 --threads 4 --num_gpus 2
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset rgbt210 --runid 15 --threads 4 --num_gpus 2

# echo '-----------------------------  Analysis ODTrack_256 for RGBT234 -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt210 --dataset_name rgbt234 --runid 15

# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name rgbt210 --runid 15

