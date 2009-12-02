
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =	 
LOCDIR          = src/dm/ao/examples/tutorials/
EXAMPLESC       = ex2.c
EXAMPLESF       = 
MANSEC          = AO

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o
#----------------------------------------------------------------------
runex2:
	-@${MPIEXEC} -n 1 ./ex2  >\
	   ex2_1.tmp 2>&1;   \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp

runex2_2:
	-@${MPIEXEC} -n 2 ./ex2  >\
	   ex2_2.tmp 2>&1;   \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp

TESTEXAMPLES_C		  = 
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C_NOCOMPLEX  = ex2.PETSc ex2.rm 
TESTEXAMPLES_PARMETIS	  = ex2.PETSc runex2 runex2_2 ex2.rm 

include ${PETSC_DIR}/conf/test
