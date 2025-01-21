export CUDA_VISIBLE_DEVICES=${1}
model_name=TSMixer
model_id_name=ETTm2
model_t=${10}
pred_len=${2}
seq_len=${11}
dataset=ETTm2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate ${3} \
  --lradj ${4} \
  --d_model ${5} \
  --train_epochs ${6} \
  --norm ${7} \
  --alpha ${8} \
  --beta ${9} \
  --model_t "$model_t" \
  --distillation