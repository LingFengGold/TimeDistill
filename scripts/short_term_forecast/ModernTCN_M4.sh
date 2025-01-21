export CUDA_VISIBLE_DEVICES=${1}

if [ ! -d "./logs/ShortForecasting" ]; then
    mkdir -p ./logs/ShortForecasting
fi

model_name=ModernTCN
root_path='./dataset/m4'
train_epochs=100
patience=10
itr=1
des='Exp'
loss='SMAPE'

seasonal_patterns=${2}

if [ "$seasonal_patterns" == "Yearly" ]; then
    ffn_ratio=4
    patch_size=3
    patch_stride=1
    num_blocks=1
    dims="2048 2048 2048 2048"
    batch_size=128
    learning_rate=0.0005
    dropout=0.5
    lradj='type1'
elif [ "$seasonal_patterns" == "Monthly" ]; then
    ffn_ratio=1
    patch_size=8
    patch_stride=4
    num_blocks=2
    dims="2048 2048 2048 2048"
    batch_size=128
    learning_rate=0.001
    dropout=0.0
    lradj='type1'
elif [ "$seasonal_patterns" == "Quarterly" ]; then
    ffn_ratio=4
    patch_size=2
    patch_stride=1
    num_blocks=2
    dims="2048 2048 2048 2048"
    batch_size=64
    learning_rate=0.0005
    dropout=0.0
    lradj='type1'
elif [ "$seasonal_patterns" == "Weekly" ]; then
    ffn_ratio=1
    patch_size=8
    patch_stride=4
    num_blocks=2
    dims="1024 1024 1024 1024"
    batch_size=32
    learning_rate=0.0001
    dropout=0.5
    lradj='type4'
elif [ "$seasonal_patterns" == "Daily" ]; then
    ffn_ratio=1
    patch_size=8
    patch_stride=4
    num_blocks=2
    dims="1024 1024 1024 1024"
    batch_size=32
    learning_rate=0.001
    dropout=0.0
    lradj='type1'
elif [ "$seasonal_patterns" == "Hourly" ]; then
    ffn_ratio=1
    patch_size=16
    patch_stride=8
    num_blocks=1
    dims="32 32 32 32"
    batch_size=8
    learning_rate=0.001
    dropout=0.3
    lradj='type1'
fi

model_id=m4_$seasonal_patterns

python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --seasonal_patterns $seasonal_patterns \
    --model_id $model_id \
    --model $model_name \
    --data m4 \
    --enc_in 1 \
    --loss $loss \
    --ffn_ratio $ffn_ratio \
    --patch_size $patch_size \
    --patch_stride $patch_stride \
    --num_blocks $num_blocks \
    --large_size 51 \
    --small_size 5 \
    --dims $dims \
    --head_dropout 0.0 \
    --dropout $dropout \
    --itr $itr \
    --learning_rate $learning_rate \
    --batch_size $batch_size \
    --train_epochs $train_epochs \
    --patience $patience \
    --des $des \
    --use_multi_scale False \
    --small_kernel_merged False \
    --lradj $lradj
