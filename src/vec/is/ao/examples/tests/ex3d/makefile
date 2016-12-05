
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/is/ao/examples/tests/ex3d/
EXAMPLESC       = ex3.cxx
EXAMPLESF       =
MANSEC          = Vec
SUBMANSEC       = AO

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_VEC_LIB}
	${RM} -f ex3.o

#-------------------------------------------------------------------
runex3:
	-@${MPIEXEC} -n 12 ./ex3 -datafiles ${DATAFILESPATH}/ao > ex3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp
runex3_2:
	-@${MPIEXEC} -n 12 ./ex3 -ao_type basic -datafiles ${DATAFILESPATH}/ao > ex3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp
runex3_3:
	-@${MPIEXEC} -n 30 ./ex3 -datafiles ${DATAFILESPATH}/ao > ex3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_3.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp
runex3_4:
	-@${MPIEXEC} -n 30 ./ex3 -ao_type basic -datafiles ${DATAFILESPATH}/ao > ex3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_3.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp

TESTEXAMPLES_C		  =
TESTEXAMPLES_C_X	  =
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_C_X_MPIUNI =
TESTEXAMPLES_C3		  =
TESTEXAMPLES_CXX_DATAFILESPATH = ex3.PETSc runex3 runex3_2 runex3_3 runex3_4 ex3.rm

include ${PETSC_DIR}/lib/petsc/conf/test
