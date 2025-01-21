export CUDA_VISIBLE_DEVICES=${1}

model_name=LightTS
model_id_name=ETTh2
model_t=${10}
pred_len=${2}
seq_len=${11}
dataset=ETTh2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
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