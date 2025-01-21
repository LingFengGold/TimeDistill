export CUDA_VISIBLE_DEVICES=${1}

model_name=TimeMixer

e_layers=4
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.01
d_model=32
batch_size=16
seasonal_patterns=${2}

# 使用 case 语句来判断季节性模式并设置 d_ff 参数
case $seasonal_patterns in
  'Monthly' | 'Yearly' | 'Weekly' | 'Hourly')
    d_ff=32
    ;;
  'Daily')
    d_ff=16
    ;;
  'Quarterly')
    d_ff=64
    ;;
  *)
    echo "Invalid seasonal pattern: $seasonal_patterns"
    exit 1
    ;;
esac

# 运行训练脚本
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns $seasonal_patterns \
  --model_id m4_$seasonal_patterns \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --d_model $d_model \
  --d_ff $d_ff \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --train_epochs 50 \
  --patience 20 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE'

