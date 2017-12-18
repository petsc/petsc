
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/power_grid/
EXAMPLESC       = ex1.c ex2.c ex3.c ex3adj.c ex3opt.c ex3opt_fd.c ex5.c \
                ex6.c ex7.c ex8.c ex9adj.c ex9.c ex9opt.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            = stability_9bus
CLEANFILES      = SA-data/*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_TS_LIB}
	${RM} ex2.o

ex2f: ex2f.o  chkopts
	-${FLINKER} -o ex2f ex2f.o   ${PETSC_TS_LIB}
	${RM} ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex3fwd: ex3fwd.o  chkopts
	-${CLINKER} -o ex3fwd ex3fwd.o  ${PETSC_TS_LIB}

ex3adj: ex3adj.o  chkopts
	-${CLINKER} -o ex3adj ex3adj.o  ${PETSC_TS_LIB}
	${RM} ex3adj.o

ex3opt: ex3opt.o  chkopts
	-${CLINKER} -o ex3opt ex3opt.o  ${PETSC_LIB}
	${RM} ex3opt.o

ex3opt_fd: ex3opt_fd.o  chkopts
	-${CLINKER} -o ex3opt_fd ex3opt_fd.o  ${PETSC_LIB}
	${RM} ex3opt_fd.o

ex3opt_fwd: ex3opt_fwd.o  chkopts
	-${CLINKER} -o ex3opt_fwd ex3opt_fwd.o  ${PETSC_LIB}

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_TS_LIB}
	${RM} ex9.o

ex9adj: ex9adj.o  chkopts
	-${CLINKER} -o ex9adj ex9adj.o  ${PETSC_TS_LIB}
	${RM} ex9adj.o

ex9opt: ex9opt.o  chkopts
	-${CLINKER} -o ex9opt ex9opt.o  ${PETSC_LIB}
	${RM} ex9opt.o

include ${PETSC_DIR}/lib/petsc/conf/test
