CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/forest/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

runex1_moebius:
	-@${MPIEXEC} -n 3 ./ex1 -dm_type p4est -dm_forest_topology moebius -dm_view vtk:moebius.vtu
runex1_shell:
	-@${MPIEXEC} -n 3 ./ex1 -dm_type p8est -dm_forest_topology shell -dm_view vtk:shell.vtu
runex1_brick:
	-@${MPIEXEC} -n 3 ./ex1 -dm_type p8est -dm_forest_topology brick -dm_p4est_brick_size 2,3,5 -dm_view vtk:brick.vtu

#--------------------------------------------------------------------------

include ${PETSC_DIR}/lib/petsc/conf/test
