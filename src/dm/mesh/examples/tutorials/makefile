
CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve
FFLAGS	        =
CPPFLAGS        = -DALE_USE_LOGGING -DALE_LOGGING_USE_EVENTS -DALE_HAVE_CXX_ABI -DALE_USE_DEBUGGING
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

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS}

debugex1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex1:
	-@valgrind --tool=memcheck ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS}

runex1_pylith:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -file_type pylith -output_local 1 -base_file /PETSc3/geoframe/packages/lithomop/lithomop3d/examples/lintet/bm1b/bm1b ${EXTRA_ARGS}

debugex1_pylith:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -file_type pylith -output_local 1 -base_file /PETSc3/geoframe/packages/lithomop/lithomop3d/examples/lintet/bm1b/bm1b -start_in_debugger ${EXTRA_ARGS}

runex1_2:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 2 -use_zero_base 1 -base_file data/ex1_2d ${EXTRA_ARGS}

runex1_3:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -use_zero_base 1 -base_file data/ex1_3d ${EXTRA_ARGS}

runex2:
	-@${MPIRUN} -np ${NP} ./ex2 -dim 2 ${EXTRA_ARGS}

valex2:
	-@valgrind --tool=memcheck ./ex2 -dim 2 -refinement_limit 0.25 ${EXTRA_ARGS}

debugex2:
	-@${MPIRUN} -np ${NP} ./ex2 -dim 2 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

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
