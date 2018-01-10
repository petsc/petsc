
CFLAGS 	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex4.c ex5.c ex8.c ex9.c ex10.c ex11.c ex12.c ex15.c ex16.c ex17.c
EXAMPLESF       = ex6f.F90
MANSEC          = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER}  -o ex3 ex3.o ${PETSC_MAT_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER}  -o ex4 ex4.o ${PETSC_MAT_LIB}
	${RM} ex4.o

ex2_kds: ex2_kds.o  chkopts
	-${CLINKER}  -o ex2_kds ex2_kds.o ${PETSC_MAT_LIB}
	${RM} ex2_kds.o

ex5: ex5.o  chkopts
	-${CLINKER}  -o ex5 ex5.o ${PETSC_MAT_LIB}
	${RM} ex5.o

ex6f: ex6f.o  chkopts
	-${CLINKER}  -o ex6f ex6f.o ${PETSC_MAT_LIB}
	${RM} ex6f.o

ex8: ex8.o  chkopts
	-${CLINKER}  -o ex8 ex8.o ${PETSC_MAT_LIB}
	${RM} ex8.o
ex9: ex9.o  chkopts
	-${CLINKER}  -o ex9 ex9.o ${PETSC_MAT_LIB}
	${RM} ex9.o
ex10: ex10.o  chkopts
	-${CLINKER}  -o ex10 ex10.o ${PETSC_MAT_LIB}
	${RM} ex10.o
ex11: ex11.o  chkopts
	-${CLINKER}  -o ex11 ex11.o ${PETSC_MAT_LIB}
	${RM} ex11.o
ex12: ex12.o  chkopts
	-${CLINKER}  -o ex12 ex12.o ${PETSC_MAT_LIB}
	${RM} ex12.o
ex13: ex13.o  chkopts
	-${CLINKER}  -o ex13 ex13.o ${PETSC_MAT_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER}  -o ex14 ex14.o ${PETSC_DM_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER}  -o ex15 ex15.o ${PETSC_MAT_LIB}
	${RM} ex15.o

ex16: ex16.o  chkopts
	-${CLINKER}  -o ex16 ex16.o ${PETSC_MAT_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER}  -o ex17 ex17.o ${PETSC_MAT_LIB}
	${RM} ex17.o

include ${PETSC_DIR}/lib/petsc/conf/test

