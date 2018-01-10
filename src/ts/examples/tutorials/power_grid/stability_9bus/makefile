
CFLAGS	        = -g
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/power_grid/stability_9bus/
EXAMPLESC       = ex9bus.c ex9busadj.c ex9busopt.c ex9busoptfd.c ex9busdmnetwork.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            =
CLEANFILES      = SA-data/*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex9bus: ex9bus.o  chkopts
	-${CLINKER} -o ex9bus ex9bus.o  ${PETSC_TS_LIB}
	${RM} ex9bus.o

ex9busadj: ex9busadj.o  chkopts
	-${CLINKER} -o ex9busadj ex9busadj.o  ${PETSC_TS_LIB}
	${RM} ex9busadj.o

ex9busopt: ex9busopt.o  chkopts
	-${CLINKER} -o ex9busopt ex9busopt.o  ${PETSC_LIB}

ex9busoptfd: ex9busoptfd.o  chkopts
	-${CLINKER} -o ex9busoptfd ex9busoptfd.o  ${PETSC_LIB}

ex9busdmnetwork: ex9busdmnetwork.o  chkopts
	-${CLINKER} -o ex9busdmnetwork ex9busdmnetwork.o  ${PETSC_TS_LIB}


include ${PETSC_DIR}/lib/petsc/conf/test
