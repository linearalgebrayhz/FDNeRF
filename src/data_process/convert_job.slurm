#!/bin/bash
#SBATCH --job-name=fdnerf        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1          
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS) 
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=hy4522@princeton.edu

# export TORCH_HOME=/scratch/network/hy4522/.cache/torch

module purge
module load anaconda3/2024.10
conda activate facenerf
cd ../..

# python ./src/data_process/video_preprocess.py

# nvcc --version
# nvidia-smi

python -u ./src/data_process/fdnerf_converter.py \
    --input_dir /scratch/network/hy4522/FDNeRF_data/mp4/exp_result \
    --output_dir /scratch/network/hy4522/FDNeRF_data/FDNeRF_converted \
    --generate_3dmm \

# nvidia-smi