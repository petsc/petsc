
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

include ${PETSC_DIR}/lib/petsc/conf/test
