# nohup bash odtrack_train.sh >>./log/train_odtrack256_ab_layer_12.out&
# sh test_xyl.sh > xyl-test.log 2>&1

# chmod +x test_yk.sh
# ./test_yk.sh

gpu="0"
# gpu="4,5,6,7"

# echo '-----------------------------  Test ODTrack_256  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 15 --threads 4 --num_gpus 4
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt234 --runid 15 --threads 4 --num_gpus 4
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset rgbt210 --runid 15 --threads 4 --num_gpus 4
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset VTUAV_test --runid 15 --threads 4 --num_gpus 4


# echo '-----------------------------  Analysis ODTrack_256  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 15
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name rgbt234 --runid 15
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name rgbt210 --runid 15
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name VTUAV_test --runid 15

# echo '------ 384版本，需要更改 lib/models/odtrack/odtrack.py 102行！！ --------------'
# echo '-----------------------------  Test ODTrack_384  -----------------------------'
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset lasher_test --runid 12 --threads 4 --num_gpus 4

# echo '-----------------------------  Analysis ODTrack_384  -----------------------------'
# python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_384_lasher --dataset_name lasher_test --runid 12

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher4rgbt234 --dataset lasher_test --runid 15 --threads 4 --num_gpus 2

# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 14 --threads 4 --num_gpus 4
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 13 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 12 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 11 --threads 4 --num_gpus 1


# python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset VTUAV_test --runid 15 --threads 4 --num_gpus 1

#-------------------------------------------------256 VTUAV test_________________________________________________________________________________________________________________
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset VTUAV_test --runid 15 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset VTUAV_test --runid 14 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset VTUAV_test --runid 13 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset VTUAV_test --runid 12 --threads 4 --num_gpus 1
# CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_vtuav --dataset VTUAV_test --runid 11 --threads 4 --num_gpus 1

#-------------------------------------------------256 LasHeR test_________________________________________________________________________________________________________________
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher_DWAM --dataset LasHeR_test --runid 15 --threads 4 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher_DWAM --dataset LasHeR_test --runid 14 --threads 4 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher_DWAM --dataset LasHeR_test --runid 13 --threads 4 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher_DWAM --dataset LasHeR_test --runid 12 --threads 4 --num_gpus 1
CUDA_VISIBLE_DEVICES=${gpu} python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher_DWAM --dataset LasHeR_test --runid 11 --threads 4 --num_gpus 1