path=best_results
mkdir -p $path

declare -A alpha_dist=(
    ["ECL"]=0.5 ["Traffic"]=1 ["Solar"]=0.1 
    ["ETTh1"]=1 ["ETTh2"]=2 ["ETTm1"]=1 
    ["ETTm2"]=1 ["Weather"]=0.5
)

declare -A beta_dist=(
    ["ECL"]=0.5 ["Traffic"]=0.1 ["Solar"]=0.1 
    ["ETTh1"]=2 ["ETTh2"]=0.1 ["ETTm1"]=1 
    ["ETTm2"]=0.1 ["Weather"]=2
)

declare -A norm_dist=(
    ["ECL"]="revin" ["Traffic"]="revin" ["Weather"]="revin" ["ETTm2"]="non-stationary"
    ["ETTh1"]="non-stationary" ["ETTh2"]="non-stationary" ["ETTm1"]="non-stationary" ["Solar"]="non-stationary"
)

method=DMLP

run_distillation() {
    local dataset=$1
    local gpu=$2
    local pred_len=$3
    local alpha=${alpha_dist[$dataset]}
    local beta=${beta_dist[$dataset]}
    local norm=${norm_dist[$dataset]}
    local learning_rate=0.01
    local lradj=type4
    local d_model=512
    if [[ "$dataset" == "ETTm1" || "$dataset" == "ETTm2" || "$dataset" == "ETTh1" || "$dataset" == "ETTh2" || "$dataset" == "Weather" ]]; then
        local train_epochs=100
    elif [[ "$dataset" == "ECL" || "$dataset" == "Solar" ]]; then
        local train_epochs=30
    elif [[ "$dataset" == "Traffic" ]]; then
        local train_epochs=10
    else
        echo "Unknown dataset: ${dataset}"
        exit 1
    fi
    local model_t=iTransformer

    echo "Running distillation for ${dataset} with pred_len=${pred_len} on GPU=${gpu}..."
    bash ./scripts/distillation/${dataset}/${method}.sh \
        ${gpu} ${pred_len} ${learning_rate} ${lradj} ${d_model} ${train_epochs} ${norm} ${alpha} ${beta} ${model_t} \
        >> ${path}/${dataset}_${pred_len}_${learning_rate}_${lradj}_${d_model}_${train_epochs}_${norm}_${alpha}_${beta}_${model_t}_${method}.out
}

for dataset in Weather ECL Solar Traffic; do
    run_distillation $dataset 0 96 &
    run_distillation $dataset 1 192 &
    run_distillation $dataset 2 336 &
    run_distillation $dataset 3 720 &
    wait 
done