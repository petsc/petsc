
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

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o  ${PETSC_DM_LIB}
	${RM} -f ex1f90.o

ex2f90: ex2f90.o  chkopts
	-${FLINKER} -o ex2f90 ex2f90.o  ${PETSC_DM_LIB}
	${RM} -f ex2f90.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} -f ex3.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} -f ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o ${PETSC_SNES_LIB}
	${RM} -f ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o ${PETSC_SNES_LIB}
	${RM} -f ex11.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o ${PETSC_SNES_LIB}
	${RM} -f ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
	${RM} -f ex13.o

ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o ${PETSC_DM_LIB}
	${RM} -f ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o ${PETSC_DM_LIB}
	${RM} -f ex15.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o ${PETSC_DM_LIB}
	${RM} -f ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o ${PETSC_DM_LIB}
	${RM} -f ex19.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o ${PETSC_DM_LIB}
	${RM} -f ex20.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o ${PETSC_DM_LIB}
	${RM} -f ex22.o


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
