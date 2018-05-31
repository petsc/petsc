CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/plex/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex5.c
EXAMPLESF       = ex1f90.F90
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#--------------------------------------------------------------------------

runex1:
	-@${MPIEXEC} -n 2 ./ex1 -dim 3

runex1f90:
	-@${MPIEXEC} -n 2 ./ex1 -dim 2

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -dim 3 -dm_refine 2 -dm_view hdf5:ex2.h5
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex2.h5

runex5:
	-@${MPIEXEC} -n 2 ./ex5 -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/square.msh -new_dm_view

include ${PETSC_DIR}/lib/petsc/conf/test
