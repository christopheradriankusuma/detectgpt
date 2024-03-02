#!/bin/sh
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adrian@comp.nus.edu.sg
#SBATCH --partition=long
#SBATCH --time=2880
# SBATCH --gpus=t4:1

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum squad writing"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B gpt-j-6B gpt-neox-20b"

# # preparing dataset
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Preparing dataset ${D}_${M} ...
#     srun python fast-detect-gpt/scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M}
#   done
# done

# # White-box Setting
# echo `date`, Evaluate models in the white-box setting:

# # evaluate Fast-DetectGPT and fast baselines
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
#     srun python fast-detect-gpt/scripts/fast_detect_gpt.py --reference_model_name $M --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}

#     echo `date`, Evaluating baseline methods on ${D}_${M} ...
#     srun python fast-detect-gpt/scripts/baselines.py --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
#   done
# done

# # evaluate DNA-GPT
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
#     srun python fast-detect-gpt/scripts/dna_gpt.py --base_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
#   done
# done

# # evaluate DetectGPT and its improvement DetectLLM
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating DetectGPT on ${D}_${M} ...
#     srun python fast-detect-gpt/scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
#      # we leverage DetectGPT to generate the perturbations
#     echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
#     srun python fast-detect-gpt/scripts/detect_llm.py --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M}
#   done
# done


# Black-box Setting
echo `date`, Evaluate models in the black-box setting:
scoring_models="pythia-70m pythia-70m-dd pythia-160m pythia-160m-dd opt-125m opt-350m distilgpt2 gpt2 gpt-neo-125m"

# evaluate Fast-DetectGPT
for D in $datasets; do
  for M in $source_models; do
    M1=gpt-j-6B  # sampling model
    for M2 in $scoring_models; do
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      srun python fast-detect-gpt/scripts/fast_detect_gpt.py --reference_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done

# evaluate DetectGPT and its improvement DetectLLM
for D in $datasets; do
  for M in $source_models; do
    M1=t5-3b  # perturbation model
    for M2 in $scoring_models; do
      echo `date`, Evaluating DetectGPT on ${D}_${M}.${M1}_${M2} ...
      srun python fast-detect-gpt/scripts/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
      # we leverage DetectGPT to generate the perturbations
      echo `date`, Evaluating DetectLLM methods on ${D}_${M}.${M1}_${M2} ...
      srun python fast-detect-gpt/scripts/detect_llm.py --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}_${M}.${M1}.perturbation_100 --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done
