
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/hybrid/
EXAMPLESC       = ex1.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1adj: ex1adj.o  chkopts
	-${CLINKER} -o ex1adj ex1adj.o  ${PETSC_TS_LIB}
	${RM} ex1adj.o

ex1fd: ex1fd.o  chkopts
	-${CLINKER} -o ex1fd ex1fd.o  ${PETSC_TS_LIB}
	${RM} ex1fd.o

ex1fwd: ex1fwd.o  chkopts
	-${CLINKER} -o ex1fwd ex1fwd.o  ${PETSC_TS_LIB}
	${RM} ex1fwd.o

include ${PETSC_DIR}/lib/petsc/conf/test
