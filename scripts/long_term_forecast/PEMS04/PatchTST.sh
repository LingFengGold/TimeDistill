export CUDA_VISIBLE_DEVICES=${1}

seq_len=96
model_name=PatchTST

root_path_name=./dataset/PEMS/
data_path_name=PEMS04.npz
model_id_name=PEMS04
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
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --learning_rate 0.0005 \
  --itr 1 \
  --use_norm 0