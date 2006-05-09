
CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS	        =
CPPFLAGS        = -DALE_USE_DEBUGGING -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_LOGGING_USE_EVENTS # -DALE_LOGGING_USE_STAGES
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

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS}

debugex1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

valex1:
	-@${MPIRUN} -np ${NP} valgrind --tool=memcheck ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d ${EXTRA_ARGS}

runex1_pylith:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -file_type pylith -output_local 1 -base_file data/bm1b/bm1b ${EXTRA_ARGS}

debugex1_pylith:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -file_type pylith -output_local 1 -base_file data/bm1b/bm1b -display ${DISPLAY} -start_in_debugger ${EXTRA_ARGS}

runex1_pylith2:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -file_type pylith -interpolate 1 -output_local 1 -base_file data/bm1m/bm1m ${EXTRA_ARGS}

debugex1_pylith2:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -file_type pylith -interpolate 1 -output_local 1 -base_file data/bm1m/bm1m ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

runex1_2:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 2 -use_zero_base 1 -base_file data/ex1_2d ${EXTRA_ARGS}

runex1_3:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -use_zero_base 1 -base_file data/ex1_3d ${EXTRA_ARGS}

runex2:
	-@${MPIRUN} -np ${NP} ./ex2 -dim 2 ${EXTRA_ARGS}

valex2:
	-@${MPIRUN} -np ${NP} valgrind --tool=memcheck ./ex2 -dim 2 ${EXTRA_ARGS}

valex2_3:
	-@valgrind --tool=memcheck ./ex2 -dim 3 -refinement_limit 0.05 ${EXTRA_ARGS}

debugex2:
	-@${MPIRUN} -np ${NP} ./ex2 -dim 2 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

runex2_3:
	-@${MPIRUN} -np ${NP} ./ex2 -dim 3 ${EXTRA_ARGS}

runex3:
	-@${MPIRUN} -np 3 ./ex3 -mat_view  > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
testex1_1:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 1 -use_zero_base 1 -base_file data/ex1_1d > /dev/null 2>&1; \
    if (${DIFF} data/ex1_1d.lcon testMesh.lcon) then true; \
	else echo "Possible problem with ex1_1, diffs above"; fi; \
    if (${DIFF} data/ex1_1d.nodes testMesh.nodes) then true; \
	else echo "Possible problem with ex1_1, diffs above"; fi

testex1_2:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 2 -use_zero_base 1 -base_file data/ex1_2d > /dev/null 2>&1; \
    if (${DIFF} data/ex1_2d.lcon testMesh.lcon) then true; \
	else echo "Possible problem with ex1_2, diffs above"; fi; \
    if (${DIFF} data/ex1_2d.nodes testMesh.nodes) then true; \
	else echo "Possible problem with ex1_2, diffs above"; fi

testex1_3:
	-@${MPIRUN} -np ${NP} ./ex1 -dim 3 -use_zero_base 1 -base_file data/ex1_3d > /dev/null 2>&1; \
    if (${DIFF} data/ex1_3d.lcon testMesh.lcon) then true; \
	else echo "Possible problem with ex1_3, diffs above"; fi; \
    if (${DIFF} data/ex1_3d.nodes testMesh.nodes) then true; \
	else echo "Possible problem with ex1_3, diffs above"; fi

testex1_4:
	-@${MPIRUN} -np 5 ./ex1 -dim 3 -file_type pylith -output_local 1 -base_file data/bm1b/bm1b > /dev/null 2>&1; \
    if (${DIFF} data/bm1b/bm1b.0.connect testMesh.0.connect) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.0.coord testMesh.0.coord) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.1.connect testMesh.1.connect) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.1.coord testMesh.1.coord) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.2.connect testMesh.2.connect) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.2.coord testMesh.2.coord) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.3.connect testMesh.3.connect) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.3.coord testMesh.3.coord) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.4.connect testMesh.4.connect) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi; \
    if (${DIFF} data/bm1b/bm1b.4.coord testMesh.4.coord) then true; \
	else echo "Possible problem with ex1_4, diffs above"; fi

testex2_1:
	-@${MPIRUN} -np 1 ./ex2 -dim 2 -refinement_limit 0.0625 > /dev/null 2>&1; \
    if (${DIFF} data/ex2_1.vtk testMesh.vtk) then true; \
	else echo "Possible problem with ex2_1, diffs above"; fi

testex2_2:
	-@${MPIRUN} -np 2 ./ex2 -dim 2 -refinement_limit 0.0625 > /dev/null 2>&1; \
    if (${DIFF} data/ex2_1.vtk testMesh.vtk) then true; \
	else echo "Possible problem with ex2_1, diffs above"; fi

testex2_3:
	-@${MPIRUN} -np 3 ./ex2 -dim 2 -refinement_limit 0.0625 > /dev/null 2>&1; \
    if (${DIFF} data/ex2_1.vtk testMesh.vtk) then true; \
	else echo "Possible problem with ex2_1, diffs above"; fi


runex4:
	-@${MPIRUN} -np ${NP} ./ex4 ${EXTRA_ARGS}
debugex4:
	-@${MPIRUN} -np ${NP} ./ex4 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}


TESTEXAMPLES_SIEVE 		  = ex1.PETSc testex1_1 testex1_2 testex1_3 testex1_4 ex1.rm ex2.PETSc testex2_1 testex2_2 testex2_3 ex2.rm
TESTEXAMPLES_C  		  = 
TESTEXAMPLES_C_X11  	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_F90          = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_C_NOCOMPLEX  = 

include ${PETSC_DIR}/bmake/common/test
