# !/bin/bash
# python3 train_costmap_wf.py --exp-prefix 2_1_test_ --batch-size 1 --test-batch-size 1 --worker-num 4 --test --test-traj --val-file local_test.txt --stride 1 --skip 0 --crop-num 2 --data-root /home/wenshan/tmp/arl_data/full_trajs --load-model --model-name 2_1_costnet_20000.pkl

# python3 train_costmap_wf.py --exp-prefix 2_1_test_ --batch-size 1 --test-batch-size 1 --worker-num 4 --test --test-traj --val-file rough_rider_test.txt --stride 1 --skip 0 --crop-num 2 --data-root /cairo/arl_bag_files/SARA/2022_05_31_trajs --load-model --model-name 2_1_costnet_20000.pkl --test-num 418

# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 4 --test --test-traj --val-file rough_rider_test.txt --stride 1 --skip 0 --crop-num 2 --data-root /cairo/arl_bag_files/SARA/2022_05_31_trajs --load-model --model-name 2_7_costnet_10000.pkl --network 2 --net-config 1 --out-vid-file 20220531_test_0.mp4

# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 4 --test --test-traj --val-file rough_rider_test_1.txt --stride 1 --skip 0 --crop-num 2 --data-root /cairo/arl_bag_files/SARA/2022_05_31_trajs --load-model --model-name 2_7_costnet_10000.pkl --network 2 --net-config 1 --out-vid-file 20220531_test_1.mp4

# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 4 --test --test-traj --val-file rough_rider_test_2.txt --stride 1 --skip 0 --crop-num 2 --data-root /cairo/arl_bag_files/SARA/2022_05_31_trajs --load-model --model-name 2_7_costnet_10000.pkl --network 2 --net-config 1 --out-vid-file 20220531_test_2.mp4

# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 4 --test --test-traj --val-file rough_rider_test_3.txt --stride 1 --skip 0 --crop-num 2 --data-root /cairo/arl_bag_files/SARA/2022_05_31_trajs --load-model --model-name 2_7_costnet_10000.pkl --network 2 --net-config 1 --out-vid-file 20220531_test_3.mp4



# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_uniform_gravel_low_0.txt --stride 1 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_9_costnet_10000.pkl --network 2 --net-config 1 --out-vid-file arl_uniform_gravel_low_0.mp4

python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_vegetation_low.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_10_2_costnet_2000.pkl --network 2 --net-config 1 --out-vid-file arl_vegetation_low.mp4

python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_woods_loop_low.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_10_2_costnet_2000.pkl --network 2 --net-config 1 --out-vid-file arl_woods_loop_low.mp4

# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_smooth_dirt_low.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_10_2_costnet_2000.pkl --network 2 --net-config 1 --out-vid-file arl_smooth_dirt_low.mp4

# python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_woods_hill_loop_low2.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_10_2_costnet_2000.pkl --network 2 --net-config 1 --out-vid-file arl_woods_hill_loop_low2.mp4

python3 train_costmap_wf.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_uniform_gravel_low_0.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_10_2_costnet_2000.pkl --network 2 --net-config 1 --out-vid-file arl_uniform_gravel_low_0.mp4

# Talay Use (For Stepping Through the Code)
# python3 train_costmap_wf_test.py --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_vegetation_low.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name 2_7_costnet_20000.pkl --network 2 --net-config 1

# Talay Use (For Visualizing Model)
python3 train_costmap_wf_uncertainty.py --exp-prefix test_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj --val-file arl_20220922_vegetation_low.txt --stride 6 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --load-model --model-name t_7_costnet_5000.pkl --network 2 --net-config 0 --out-vid-file arl_vegetation_low.mp4

# Talay Final Script (For Uncertainty Estimation)
python3 train_costmap_wf_uncertainty.py --exp-prefix t1_un_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj-uncertainty --val-file arl_20220922_woods_loop_low.txt --stride 3 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --ensemble --ensemble-length 4 --load-model --ensemble-model-names t_6_costnet_2500.pkl t_7_costnet_5000.pkl t_8_costnet_5000.pkl t_9_costnet_2500.pkl --network 2 --net-config 0 --out-vid-file t1_test_uncertainty.mp4
python3 train_costmap_wf_uncertainty.py --exp-prefix t2_un_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj-uncertainty --val-file arl_20220922_uniform_gravel_low_0.txt --stride 1 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --ensemble --ensemble-length 4 --load-model --ensemble-model-names t_6_costnet_2500.pkl t_7_costnet_5000.pkl t_8_costnet_5000.pkl t_9_costnet_2500.pkl --network 2 --net-config 0 --out-vid-file t2_test_uncertainty.mp4
python3 train_costmap_wf_uncertainty.py --exp-prefix t3_un_ --batch-size 1 --test-batch-size 1 --worker-num 1 --test --test-traj-uncertainty --val-file arl_20220922_woods_hill_loop_low2.txt --stride 1 --skip 0 --crop-num 2 --data-root /project/learningphysics/arl_20220922_traj --ensemble --ensemble-length 5 --load-model --ensemble-model-names t_6_costnet_2500.pkl t_7_costnet_5000.pkl t_8_costnet_5000.pkl t_9_costnet_2500.pkl t_10_costnet_2500.pkl --network 2 --net-config 0 --out-vid-file t3_test_uncertainty.mp4