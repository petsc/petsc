
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/snes/examples/tests/
EXAMPLESC       = ex1.c  ex7.c ex17.c ex68.c ex69.c
EXAMPLESF       = ex1f.F90 ex12f.F ex18f90.F90
DIRS	        =
MANSEC          = SNES

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_SNES_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SNES_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_SNES_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_SNES_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_SNES_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_SNES_LIB}
	${RM} ex7.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${FLINKER} -o ex10 ex10.o ${PETSC_SNES_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o ${PETSC_SNES_LIB}
	${RM} ex11.o

ex12f: ex12f.o  chkopts
	-${FLINKER} -o ex12f ex12f.o ${PETSC_SNES_LIB}
	${RM} ex12f.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
	${RM} ex13.o

ex14f: ex14f.o  chkopts
	-${FLINKER} -o ex14f ex14f.o ${PETSC_SNES_LIB}
	${RM} ex14f.o

ex17: ex17.o chkopts
	-${CLINKER} -o ex17 ex17.o ${PETSC_SNES_LIB}
	${RM} ex17.o

ex18f90: ex18f90.o chkopts
	-${FLINKER} -o ex18f90 ex18f90.o ${PETSC_SNES_LIB}
	${RM} ex18f90.o

ex68: ex68.o chkopts
	-${CLINKER} -o ex68 ex68.o ${PETSC_SNES_LIB}
	${RM} ex68.o

ex69: ex69.o chkopts
	-${CLINKER} -o ex69 ex69.o ${PETSC_SNES_LIB}
	${RM} ex69.o

include ${PETSC_DIR}/lib/petsc/conf/test
