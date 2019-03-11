
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/swarm/examples/tests/
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = DM

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

NP=1
N=1
P=1
ORDER=2
VALGRIND_PROG=valgrind
VALGRIND_ARGS=--dsymutil=yes --leak-check=full --gen-suppressions=no --num-callers=20 --error-limit=no
VAL=$(VALGRIND_PROG) $(VALGRIND_ARGS)

run:
	-${MPIEXEC} -n ${NP} ./ex2 -dim 2 -pc_type lu -faces ${N} -particles_cell ${P} -petscspace_order ${ORDER} -petscfe_default_quadrature_order ${ORDER} -dm_view hdf5:sol.h5 -fhat_view hdf5:sol.h5::append -rhs_view hdf5:sol.h5::append -simplex 0 -function linear -mesh_perturbation 1.e-21 -particle_perturbation 1.e-22
	-@${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py sol.h5

# -M_p_view ascii:f.m:ascii_matlab

#-petscfe_default_quadrature_order 4

val:
	-${MPIEXEC} -n ${NP} $(VAL) ./ex2 -dim 2 -pc_type lu -faces ${N} -particles_cell ${P} -sw_view -petscspace_order ${ORDER} -petscfe_default_quadrature_order ${ORDER} -dm_view hdf5:sol.h5 -fhat_view hdf5:sol.h5::append -simplex 0 -function linear

lldb:
	-@lldb ./ex2 -- -dim 2 -dim 2 -faces 1 -pc_type lu -dm_view -sw_view -petscspace_order ${ORDER} -petscfe_default_quadrature_order ${ORDER}
