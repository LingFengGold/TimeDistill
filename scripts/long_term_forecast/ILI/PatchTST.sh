#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

seq_len=36
label_len=18
model_name=PatchTST
root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=ILI
data_name=custom
factor=3
enc_in=7
dec_in=7
c_out=7
des='Exp'
itr=1
e_layers=4
d_layers=1

# pred_len is passed as the second argument
pred_len=${2}

# Set specific parameters for different pred_len values
if [ $pred_len -eq 24 ]; then
  n_heads=4
  d_model=1024
elif [ $pred_len -eq 36 ]; then
  n_heads=4
  d_model=2048
elif [ $pred_len -eq 48 ]; then
  n_heads=4
  d_model=2048
elif [ $pred_len -eq 60 ]; then
  n_heads=16
  d_model=2048
else
  echo "Invalid pred_len value"
  exit 1
fi

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --d_model $d_model \
  --itr $itr
