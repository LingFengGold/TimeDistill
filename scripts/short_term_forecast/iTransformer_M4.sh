export CUDA_VISIBLE_DEVICES=${1}

model_name=iTransformer
d_model=512
e_layers=2
d_layers=1
factor=3
enc_in=1
dec_in=1
c_out=1
batch_size=16
learning_rate=0.001
loss='SMAPE'
des='Exp'
itr=1
root_path='./dataset/m4'
train_epochs=50
patience=20

seasonal_patterns=${2}
model_id=m4_$seasonal_patterns

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns $seasonal_patterns \
  --model_id $model_id \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --batch_size $batch_size \
  --d_model $d_model \
  --des $des \
  --itr $itr \
  --learning_rate $learning_rate \
  --loss $loss \
  --train_epochs $train_epochs \
  --patience $patience

