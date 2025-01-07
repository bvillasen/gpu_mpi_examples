#!/bin/bash
#SBATCH -p MI300
#SBATCH -J all_reduce
#SBATCH --time=0:00:00
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --threads-per-core=1
#SBATCH -e job_error.log
#SBATCH -o job_output.log
#SBATCH --exclusive

echo "SLUM_NODES=$SLURM_NNODES  NODE_LIST:$SLURM_NODELIST"
echo "Starting SLURM job. $(date)"

BASE_DIR=$HOME/gpu_mpi_examples

SYSTEM=lockhart-mi300a source ${BASE_DIR}/set_env.sh

EXEC=${BASE_DIR}/all_reduce/gpu_all_reduce_mi300a

GPU_AFFINITY=${BASE_DIR}/scripts/slurm_affinity_mi300a.sh

srun -n 4 ${GPU_AFFINITY} ${EXEC}

echo "Finished SLURM job. $(date)"




