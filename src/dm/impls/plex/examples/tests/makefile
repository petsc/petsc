
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/plex/examples/tests/
EXAMPLESC       = ex1.c ex3.c ex9.c ex15.c
EXAMPLESF       = ex1f90.F90 ex2f90.F90
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
#--------------------------------------------------------------------------
runex3_nonconforming_simplex_2_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 2 -simplex 1 -dim 2 -dm_view vtk:nonconformal_simplex_2.vtu:vtk_vtu
runex3_nonconforming_tensor_2_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 2 -simplex 0 -dim 2 -dm_view vtk:nonconformal_tensor_2.vtu:vtk_vtu
runex3_nonconforming_simplex_3_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 3 -simplex 1 -dim 3 -dm_view vtk:nonconformal_simplex_3.vtu:vtk_vtu
runex3_nonconforming_tensor_3_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 3 -simplex 0 -dim 3 -dm_view vtk:nonconformal_tensor_3.vtu:vtk_vtu

include ${PETSC_DIR}/lib/petsc/conf/test
