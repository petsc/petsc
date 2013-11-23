/soft/apps/packages/mpich-gm-1.2.6..13b-intel-8.1-2/bin/mpirun -np 64 -machinefile ${PBS_NODEFILE} /home/sarich/working/ptho/compute_energy_101  >> computelog

#/sandbox/sarich/petsc-dev/linux-gnu-cxx-opt/bin/mpiexec -np 4 /sandbox/sarich/ptho/compute_energy_101 >> computeh

#!/bin/sh
#PBS -l nodes=32
#PBS -l walltime=21:30:00

#export TAO_DIR=/home/sarich/software/tao-1.9
#export PETSC_DIR=/soft/apps/packages/petsc-packages/petsc-2.3.3
#export PETSC_ARCH=linux-rhAS3-intel81-cxx-opt

#export LD_LIBRARY_PATH=/lib:/usr/lib:$LD_LIBRARY_PATH
#cd /home/sarich/working/ptho/matplay
