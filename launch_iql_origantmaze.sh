export PYTHONPATH=$PYTHONPATH:/home/asap7772/antmaze_gen/
export PYTHONPATH=$PYTHONPATH:/home/asap7772/harness-offline-rl/implicit_q_learning
export PYTHONPATH=$PYTHONPATH:/home/asap7772/harness-offline-rl/
gpus=(0 1 2 3 4 5 6 7)
samplers=(AW-0.1 RW-0.1)
seeds=(100 200 300 400)
envs=('antmaze-medium-play-v2' 'antmaze-medium-diverse-v2' 'antmaze-large-play-v2' 'antmaze-large-diverse-v2')
exp_num=0
exp_cutoff=32

for env in ${envs[@]}; do
for seed in ${seeds[@]}; do
for sampler in ${samplers[@]}; do
    gpu=${gpus[$exp_num % ${#gpus[@]}]}
    exp_num=$((exp_num+1))
    now=$(date +"%Y%m%d_%H%M%S")

    echo "Running Experiment $exp_num on env: $env, seed: $seed, sampler: $sampler, gpu: $gpu"

    command="XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_offline.py \
    --project offline-subopt-iql \
    --track \
    --env_name $env \
    --config configs/antmaze_config.py \
    --sampler $sampler \
    --seed $seed \
    --save_dir ./results/IQL/$env/$sampler/$seed/$now &"

    echo $command
    eval $command

    if [[ $exp_num -ge $exp_cutoff ]]; then
        exit 0
    fi
    sleep 30
done
done
done