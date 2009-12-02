
CFLAGS	        =
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/viewer/examples/tutorials/
EXAMPLESC       = 
EXAMPLESF       = ex1f90.F90
MANSEC          = Viewer

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SYS_LIB}
	${RM} ex1.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_SYS_LIB}
	${RM} ex1f90.o

runex1f90:
	-@${MPIEXEC} -n 1 ./ex1f90 > ex1f90_1.tmp 5>&1; \
	   if (${DIFF} output/ex1f90_1.out ex1f90_1.tmp) then true; \
	   else echo "Possible problem with ex1f90_1, diffs above"; fi; \
	   ${RM} -f ex1f90_1.tmp 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_C_X11_MPIUNI = 

include ${PETSC_DIR}/conf/test

