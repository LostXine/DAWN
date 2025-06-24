# python dawn/data/calvin/pre2.py \
# --data_path /home/nero/Robotics/DAWN/data/calvin/dataset_opt 

# python dawn/data/calvin/preprocess.py \
# --data_path data/calvin/datasets/calvin_debug_dataset \
# --output_path data/calvin/dataset_opt/calvin_debug_dataset \
# --num_workers 8

python dawn/data/calvin/preprocess.py \
--data_path data/calvin/datasets/task_ABC_D \
--output_path data/calvin/dataset_opt/task_ABC_D \
--num_workers 8

python dawn/data/calvin/preprocess.py \
--data_path data/calvin/datasets/task_D_D \
--output_path data/calvin/dataset_opt/task_D_D \
--num_workers 8