# boostcamp_pstage10

# Run
## 1. train
python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config}

## 2. inference(submission.csv)
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root /opt/ml/data/test --data_config configs/data/taco.yaml3

