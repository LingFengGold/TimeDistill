#!/bin/bash --login

#SBATCH --exclude=lac-142,lac-343,lac-199
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G

source /mnt/home/songyu5/anaconda3/etc/profile.d/conda.sh
conda activate tsf-a100
export PATH="/mnt/home/songyu5/anaconda3/envs/tsf/bin:$PATH"

path=debug
mkdir -p $path

# method=DMLP_cd
# dataset=Traffic
# pred_len=96
# learning_rate=0.001; lradj=type1; d_model=512; train_epochs=10; norm=revin; model_t=iTransformer; gpu=0; alpha=0.5; beta=0.5;
# bash ./scripts/distillation/${dataset}/${method}.sh ${gpu} ${pred_len} ${learning_rate} ${lradj} ${d_model} ${train_epochs} ${norm} ${alpha} ${beta} ${model_t}  >> \
#                 ${path}/${dataset}_${pred_len}_${learning_rate}_${lradj}_${d_model}_${train_epochs}_${norm}_${alpha}_${beta}_${gamma}_${model_t}_${method}.out

# method=DMLP
# dataset=Traffic
# seq_len=720
# pred_len=96
# learning_rate=0.01; lradj=type1; d_model=1024; train_epochs=10; norm=revin; gpu=0;
# bash ./scripts/long_term_forecast/${dataset}/${method}.sh ${gpu} ${pred_len} ${learning_rate} ${lradj} ${d_model} ${train_epochs} ${norm} ${seq_len}>> \
#                  ${path}/${dataset}_${seq_len}_${pred_len}_${learning_rate}_${lradj}_${d_model}_${train_epochs}_${norm}_${method}.out

method=ModernTCN
dataset=Traffic
pred_len=720
seq_len=720
gpu=0
bash ./scripts/long_term_forecast/${dataset}/${method}.sh ${gpu} ${pred_len} ${seq_len} 