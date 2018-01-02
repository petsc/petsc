
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR   = src/sys/classes/draw/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex6.c ex7.c ex9.c ex10.c ex11.c ex12.c
EXAMPLESF       = ex4f.F ex5f.F
MANSEC          = Sys
SUBMANSEC       = Draw

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex4f: ex4f.o  chkopts
	-${FLINKER} -o ex4f ex4f.o  ${PETSC_SYS_LIB}
	${RM} -f ex4f.o

ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o ${PETSC_SYS_LIB}
	${RM} -f ex5f.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_SYS_LIB}
	${RM} -f ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_SYS_LIB}
	${RM} -f ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_SYS_LIB}
	${RM} -f ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_SYS_LIB}
	${RM} -f ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_SYS_LIB}
	${RM} -f ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_SYS_LIB}
	${RM} -f ex11.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_SYS_LIB}
	${RM} -f ex12.o

include ${PETSC_DIR}/lib/petsc/conf/test
