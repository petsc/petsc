
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/sys/error/examples/tests/
EXAMPLESC        = ex1.c
EXAMPLESF        = 
MANSEC           = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_LIB}
	${RM} ex1.o

#----------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ex1 > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp

#TESTEXAMPLES_C = ex1.PETSc runex1 ex1.rm

include ${PETSC_DIR}/conf/test
