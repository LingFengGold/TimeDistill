if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=Linear
model_t=iTransformer
dataset=ETTm1
pred_len=96

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id $dataset'_'96_$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
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
  --distillation \
  --alpha 0 \
  --beta 1 \
  --gamma 0 \
  --model_t $model_t \
  2>&1 | tee logs/LongForecasting/$dataset'_'$model_name'_'$model_t'_'$dataset'_'96_$pred_len'.log'