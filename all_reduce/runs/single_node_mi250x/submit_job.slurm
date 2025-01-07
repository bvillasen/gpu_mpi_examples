#!/bin/bash
#SBATCH -J all_reduce
#SBATCH --time=0:00:00
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --threads-per-core=1
#SBATCH -e job_error.log
#SBATCH -o job_output.log
#SBATCH --exclusive

echo "SLUM_NODES=$SLURM_NNODES  NODE_LIST:$SLURM_NODELIST"
echo "Starting SLURM job. $(date)"

BASE_DIR=$HOME/gpu_mpi_examples

SYSTEM=lockhart-mi250x source ${BASE_DIR}/set_env.sh

EXEC=${BASE_DIR}/all_reduce/gpu_all_reduce_mi250x

GPU_AFFINITY="--cpu-bind=verbose --cpu-bind=mask_cpu:ff000000000000,ff00000000000000,ff0000,ff000000,ff,ff00,ff00000000,ff0000000000"

srun -n 8 ${GPU_AFFINITY} ${EXEC}

echo "Finished SLURM job. $(date)"




