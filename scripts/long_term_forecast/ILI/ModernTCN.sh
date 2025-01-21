export CUDA_VISIBLE_DEVICES=${1}

seq_len=36
model_name=ModernTCN

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
  --des 'Exp' \
  --itr 1 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --dropout 0.1 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 5 \
  --learning_rate 0.0025 \
  --lradj constant \
  --use_multi_scale False \
  --small_kernel_merged False
