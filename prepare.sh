#!/bin/sh
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adrian@comp.nus.edu.sg
#SBATCH --partition=long
#SBATCH --time=600
#SBATCH --ntasks=15
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --nodes=1-5

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum squad writing"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B"
# gpt-neox-20b gpt-j-6B

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    srun --ntasks=1 --gpus-per-task=1 --nodes=1 --mem-per-gpu=2G --exclusive python fast-detect-gpt/scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M} &
  done
done

# prepare folders
exp_path=exp_gpt3to4
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum writing pubmed"
source_models="davinci-002 gpt-3.5-turbo gpt-4"

# preparing dataset
openai_base="https://api.openai.com/v1"
openai_key="xxxxxxxx"  # replace with your own key for generating your own test set

# We follow DetectGPT settings for generating text from GPT-3
M=davinci-002
for D in $datasets; do
  echo `date`, Preparing dataset ${D} by sampling from openai/${M} ...
  srun --ntasks=1 --gpus-per-task=1 --nodes=1 --mem-per-gpu=1G --exclusive python fast-detect-gpt/scripts/data_builder.py --openai_model $M --openai_key $openai_key --openai_base $openai_base \
              --dataset $D --n_samples 150 --do_top_p --top_p 0.9 --batch_size 1 \
              --output_file $data_path/${D}_${M} &
done

# We use a temperature of 0.8 for creativity writing
for M in gpt-3.5-turbo gpt-4; do
  for D in $datasets; do
    echo `date`, Preparing dataset ${D} by sampling from openai/${M} ...
    srun --ntasks=1 --gpus-per-task=1 --nodes=1 --mem-per-gpu=1G --exclusive python fast-detect-gpt/scripts/data_builder.py --openai_model $M --openai_key $openai_key --openai_base $openai_base \
                --dataset $D --n_samples 150 --do_temperature --temperature 0.8 --batch_size 1 \
                --output_file $data_path/${D}_${M} &
  done
done
wait