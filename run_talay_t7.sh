python3 /data2/datasets/tkondhor/code/ss_costmap/train_costmap_wf.py --exp-prefix t_7_ --batch-size 100 --test-batch-size 100 --lr 0.001 --lr-decay --worker-num 8 --train-step 5000 --snapshot 1250 --data-file combine_train_crop10.txt --val-file combine_test_crop10.txt --stride 3 --skip 1 --crop-num 10 --data-root /project/learningphysics --network 2