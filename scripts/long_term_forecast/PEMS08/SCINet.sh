export CUDA_VISIBLE_DEVICES=${1}

seq_len=96
model_name=SCINet

root_path_name=./dataset/PEMS/
data_path_name=PEMS08.npz
model_id_name=PEMS08
pred_len=${2}

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len ${pred_len} \
  --e_layers 1 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --dropout 0.5 \