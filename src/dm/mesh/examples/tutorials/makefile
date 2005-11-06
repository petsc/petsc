
CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       = 
MANSEC          = DA
NP              = 1
DISPLAY         = :0.0
EXTRA_ARGS      =

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -vertex_file data/ex1conn1d.data -coord_file data/ex1coord1d.data ${EXTRA_ARGS}

debugex1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -vertex_file data/ex1conn1d.data -coord_file data/ex1coord1d.data -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex1:
	-@valgrind --tool=memcheck ./ex1 -dim 1 -use_zero_base 1 -vertex_file data/ex1conn1d.data -coord_file data/ex1coord1d.data ${EXTRA_ARGS}

runex1_2:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 2 -use_zero_base 1 -vertex_file data/ex1conn2d.data -coord_file data/ex1coord2d.data

runex1_3:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -use_zero_base 1 -vertex_file data/ex1conn3d.data -coord_file data/ex1coord3d.data

debugex1_3:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -use_zero_base 1 -vertex_file data/ex1conn3d.data -coord_file data/ex1coord3d.data -start_in_debugger -display ${DISPLAY}

runex3:
	-@${MPIRUN} -np 3 ./ex3 -mat_view  > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_F90          = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_C_NOCOMPLEX  = 
TESTEXAMPLES_13		  = 

include ${PETSC_DIR}/bmake/common/test
