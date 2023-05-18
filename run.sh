python train_costmap_wf.py --batch-size 50 --test-batch-size 50 --lr 0.001 --worker-num 2
# add velocity input
python train_costmap_wf.py --exp-prefix 1_2_ --batch-size 50 --test-batch-size 50 --lr 0.001 --worker-num 0 --train-step 5000 --snapshot 1000
# patch num 2->8
python train_costmap_wf.py --exp-prefix 1_3_ --batch-size 20 --test-batch-size 20 --lr 0.001 --worker-num 0 --train-step 5000 --snapshot 1000

python3 train_costmap_wf.py --exp-prefix 2_1_ --batch-size 100 --test-batch-size 100 --lr 0.001 --worker-num 0 --train-step 50000 --snapshot 5000 --data-file combined_train_crop1.txt --val-file combined_test_crop1.txt --stride 1 --skip 0 --crop-num 1 --data-root /project/learningphysics

# crop num = 2
python3 train_costmap_wf.py --exp-prefix 2_2_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 8 --train-step 20000 --snapshot 5000 --data-file combine_train_crop2.txt --val-file combine_test_crop2.txt --stride 1 --skip 0 --crop-num 2 --data-root /project/learningphysics

# crop num = 5
python3 train_costmap_wf.py --exp-prefix 2_3_2_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 8 --train-step 20000 --snapshot 5000 --data-file combine_train_crop5.txt --val-file combine_test_crop5.txt --stride 1 --skip 0 --crop-num 5 --data-root /project/learningphysics

# only tartandrive
python3 train_costmap_wf.py --exp-prefix 2_4_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 4 --train-step 20000 --snapshot 5000 --data-file combine_train_crop2_tartandrive.txt --val-file combine_test_crop2.txt --stride 1 --skip 0 --crop-num 2 --data-root /project/learningphysics

# two heads
python3 train_costmap_wf.py --exp-prefix 2_5_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 8 --train-step 20000 --snapshot 5000 --data-file combine_train_crop5.txt --val-file combine_test_crop5.txt --stride 1 --skip 0 --crop-num 5 --data-root /project/learningphysics --network 2

# two heads
python3 train_costmap_wf.py --exp-prefix 2_6_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 8 --train-step 20000 --snapshot 5000 --data-file combine_train_crop10.txt --val-file combine_test_crop10.txt --stride 1 --skip 0 --crop-num 10 --data-root /project/learningphysics --network 2

# smaller net + data augmentation
python3 train_costmap_wf.py --exp-prefix 2_7_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 8 --train-step 20000 --snapshot 5000 --data-file combine_train_crop10.txt --val-file combine_test_crop10.txt --stride 1 --skip 0 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1

# arl data finetune
python3 train_costmap_wf.py --exp-prefix 2_8_ --batch-size 100 --test-batch-size 100 --lr 0.0001 --lr-decay --worker-num 8 --train-step 10000 --snapshot 2000 --data-file arl_combine_train_crop20.txt --val-file arl_combine_test_crop20.txt --stride 1 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1 --load-model --model-name 2_7_costnet_10000.pkl

# arl data finetune (forgot fix layers)
python3 train_costmap_wf.py --exp-prefix 2_9_2_ --batch-size 100 --test-batch-size 100 --lr 0.0001 --lr-decay --worker-num 8 --train-step 10000 --snapshot 2000 --data-file arl_combine_train_crop20.txt --val-file arl_combine_test_crop20.txt --stride 1 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1 --load-model --model-name 2_7_costnet_10000.pkl --finetune

# fix finetune bug
python3 train_costmap_wf.py --exp-prefix 2_10_ --batch-size 100 --test-batch-size 100 --lr 0.0001 --lr-decay --worker-num 2 --train-step 10000 --snapshot 1000 --data-file arl_combine_train_crop20.txt --val-file arl_combine_test_crop20.txt --stride 1 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1 --load-model --model-name 2_7_costnet_10000.pkl --finetune

# only finetune 2 layers
python3 train_costmap_wf.py --exp-prefix 2_11_ --batch-size 32 --test-batch-size 32 --lr 0.0001 --lr-decay --worker-num 2 --train-step 10000 --snapshot 1000 --data-file arl_combine_train_crop20.txt --val-file arl_combine_test_crop20.txt --stride 1 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1 --load-model --model-name 2_7_costnet_10000.pkl --finetune

python3 train_costmap_wf.py --exp-prefix 2_12_ --batch-size 32 --test-batch-size 32 --lr 0.0001 --lr-decay --worker-num 4 --train-step 10000 --snapshot 1000 --data-file arl_combine_train_crop20.txt --val-file arl_combine_test_crop20.txt --stride 1 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1 --load-model --model-name 2_7_costnet_10000.pkl --finetune

python3 train_costmap_wf.py --exp-prefix 2_5_local_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 0 --train-step 20000 --snapshot 5000 --data-file rough_rider.txt --val-file rough_rider_test.txt --stride 1 --skip 0 --crop-num 5 --data-root /cairo/arl_bag_files/SARA/2022_05_31_trajs --network 2


python3 train_costmap_wf.py --exp-prefix 2_8_local_ --batch-size 100 --test-batch-size 100 --lr 0.0001 --lr-decay --worker-num 8 --train-step 10000 --snapshot 2000 --data-file arl_local.txt --val-file combine_test_crop10.txt --stride 1 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2 --net-config 1 --load-model --model-name 2_7_costnet_10000.pkl --finetune