path=debug
mkdir -p $path

# method=TSMixer
# dataset=ECL
# seq_len=720
# pred_len=96
# learning_rate=0.01; lradj=type1; d_model=512; train_epochs=10; norm=revin; model_t=ModernTCN; gpu=3; alpha=1; beta=1;
# bash ./scripts/distillation/${dataset}/${method}.sh ${gpu} ${pred_len} ${learning_rate} ${lradj} ${d_model} ${train_epochs} ${norm} ${alpha} ${beta} ${model_t} ${seq_len}


# method=DMLP
# dataset=Traffic
# pred_len=96
# learning_rate=0.01; lradj=type1; d_model=512; train_epochs=10; norm=revin; gpu=3;
# bash ./scripts/long_term_forecast/${dataset}/${method}.sh ${gpu} ${pred_len} ${learning_rate} ${lradj} ${d_model} ${train_epochs} ${norm}

method=TSMixer
dataset=ECL
pred_len=96
seq_len=720
gpu=2
bash ./scripts/long_term_forecast/${dataset}/${method}.sh ${gpu} ${pred_len} ${seq_len}
>> $path/${dataset}_${seq_len}_${pred_len}_${method}.out