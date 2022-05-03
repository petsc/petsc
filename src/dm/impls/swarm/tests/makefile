-include ../../../../../petscdir.mk

LOCDIR          = src/dm/impls/swarm/tests/
EXAMPLESC       = ex1.c ex2.c ex4.c ex5.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

NP=1
N=1
P=1
ORDER=2

run:
	-${MPIEXEC} -n ${NP} ./ex2 -dim 2 -pc_type lu -faces ${N} -particles_cell ${P} -petscspace_order ${ORDER} -petscfe_default_quadrature_order ${ORDER} -dm_view hdf5:sol.h5 -fhat_view hdf5:sol.h5::append -rhs_view hdf5:sol.h5::append -simplex 0 -function linear -mesh_perturbation 1.e-21 -particle_perturbation 1.e-22
	-@${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py sol.h5

