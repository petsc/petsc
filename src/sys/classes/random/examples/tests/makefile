
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/classes/random/examples/tests/
EXAMPLESC       = ex3.c
EXAMPLESF       =
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

include ${PETSC_DIR}/lib/petsc/conf/test
