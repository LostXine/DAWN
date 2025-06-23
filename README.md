# Diffusion is All We Need

## Installation
```
Will be updated!
```
## Data Preparation
```
Will be updated!
```
## Training
### Stage 1: Pixel Motion Estimation
```
accelerate launch train.py config=stage1
```
### Stage 2: Action Prediction
```
accelerate launch train.py config=stage2
```
We can easily modify the model/dataset or any parameter follow `hydra` lib. E.g.
```
accelerate launch train.py config=stage1 model=LTM_trans dataset=calvin_D_D loader.val_batch_size=8
```
## Evaluation
```
python calvin_infer.py weights=<weight_path>
```