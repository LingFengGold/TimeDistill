#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

seq_len=96
model_name=PatchTST
root_path_name=./dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
label_len=48
factor=3
enc_in=8
dec_in=8
c_out=8
des='Exp'
itr=1
e_layers=2
d_layers=1
train_epochs=10

# pred_len is passed as the second argument
pred_len=${2}

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
  --itr $itr \
  --train_epochs $train_epochs
