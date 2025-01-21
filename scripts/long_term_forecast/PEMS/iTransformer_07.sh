export CUDA_VISIBLE_DEVICES=${1}

seq_len=96
model_name=iTransformer

root_path_name=./dataset/PEMS/
data_path_name=PEMS07.npz
model_id_name=PEMS07
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
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 \
  --use_norm 0