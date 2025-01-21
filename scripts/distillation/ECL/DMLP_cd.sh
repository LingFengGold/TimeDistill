export CUDA_VISIBLE_DEVICES=${1}

model_name=DMLP_cd
model_t=${10}
pred_len=${2}
seq_len=336
dataset=ECL

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
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
  --distillation \
