export CUDA_VISIBLE_DEVICES=${1}

seq_len=36
model_name=SCINet

root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=ILI
data_name=custom
pred_len=${2}

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16