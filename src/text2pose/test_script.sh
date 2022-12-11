#!/bin/bash

##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################


###############################################################################
# SCRIPT ARGUMENTS

while getopts a:c:s: flag
do
    case "${flag}" in
        c) config=${OPTARG};; # configuration of the experiment
        s) run_id_seed=${OPTARG};; # (optional) seed value
    esac
done

# default values
: ${run_id_seed=0}

###############################################################################
# CONFIGURATION OF THE EXPERIMENT

# global/default configuration
args=(
    --model 'CondTextPoser'
    --text_encoder_name 'glovebigru_vocA1H1'
    --latentD 32
    --wloss_v2v 4 --wloss_rot 2 --wloss_jts 2
    --wloss_kld 0.2 --wloss_kldnpmul 0.02
    --lr 0.0001 --wd 0.0001
    --batch_size 32
    --seed ${run_id_seed}
)
nb_epoch=2000

# specific configuration
if [ "$config" == "gen_glovebigru_vocA1H1_dataA1" ]; then
    dataset='posescript-A1'
    ret_model_for_recall='ret_glovebigru_vocA1H1_dataA1'
    fid='ret_glovebigru_vocA1H1_dataA1'

elif [ "$config" == "gen_glovebigru_vocA1H1_dataH1" ]; then
    dataset='posescript-H1'
    ret_model_for_recall='ret_glovebigru_vocA1H1_dataH1'
    fid='ret_glovebigru_vocA1H1_dataH1'

else
    echo "Provided config (-c ${config}) is unknown."
fi
args+=(--dataset $dataset)

# utils
model_dir_path=$(python option.py "${args[@]}")
# model_path="${model_dir_path}/checkpoint_$((${nb_epoch}-1)).pth" # used for evaluation
model_path="${model_dir_path}/checkpoint_last.pth" # used for evaluation


    echo "This is our Retrieval model's score."
    bash retrieval/script_retrieval.sh -a "-eval" -s "${run_id_seed}" -c "${fid}"

    echo "This is our Generative model's score."
    python generative/our_train_eval_main.py --mode 'eval' --dataset $dataset \
    --model_path $model_path --fid $fid --split 'test'

    echo "mRecall of the result is mRecall R/G metric for generative model"
    bash retrieval/script_retrieval.sh -a "eval" -s "${run_id_seed}" -c "${ret_model_for_recall}" -j "${config}"

    echo "mRecall of the result is mRecall G/R metric for generative model"
    bash retrieval/script_retrieval.sh -a "eval" -s "${run_id_seed}" -c "${ret_model_for_recall}" -g "${config}"

