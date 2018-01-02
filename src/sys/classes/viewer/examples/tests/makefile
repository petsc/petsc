
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR  = src/sys/classes/viewer/examples/tests/
EXAMPLESC       = ex3.c ex4.c
MANSEC          = Sys
SUBMANSEC       = Viewer

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_SYS_LIB}
	${RM} -f ex5.o

include ${PETSC_DIR}/lib/petsc/conf/test

