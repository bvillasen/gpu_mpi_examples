#!/bin/bash

echo -e "Setting environment for system: ${SYSTEM}"

if [[ "${SYSTEM}" = "lockhart-mi250x" ]]; then
  module load rocm
  module load cray-mpich
  module load craype-accel-amd-gfx90a
  export MPICH_GPU_SUPPORT_ENABLED=1
  export GPU_ARCH="gfx90a"
  export GPU_TYPE="mi250x"

elif [[ "${SYSTEM}" = "lockhart-mi300a" ]]; then
  module load rocm
  module load cray-mpich
  module load craype-accel-amd-gfx942
  export MPICH_GPU_SUPPORT_ENABLED=1
  export GPU_ARCH="gfx942"
  export GPU_TYPE="mi300a"
else
  echo -e "System: ${SYSTEM} not in list of known systems. "
  return
fi