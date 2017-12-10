
CFLAGS	        =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ksp/pc/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = ex8f.F
MANSEC          = KSP
SUBMANSEC       = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_KSP_LIB}
	${RM} ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_KSP_LIB}
	${RM} ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_KSP_LIB}
	${RM} ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_KSP_LIB}
	${RM} ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_KSP_LIB}
	${RM} ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_KSP_LIB}
	${RM} ex7.o

ex8f: ex8f.o   chkopts
	-${FLINKER} -o ex8f ex8f.o ${PETSC_KSP_LIB}
	${RM} ex8f.o

include ${PETSC_DIR}/lib/petsc/conf/test


