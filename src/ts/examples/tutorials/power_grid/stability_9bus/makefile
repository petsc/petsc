
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/power_grid/stability_9bus/
EXAMPLESC       = ex9bus.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex9bus: ex9bus.o  chkopts
	-${CLINKER} -o ex9bus ex9bus.o  ${PETSC_TS_LIB}
	${RM} ex9bus.o

