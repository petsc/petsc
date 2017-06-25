
ALL:

CFLAGS	        =
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex9.c ex11.c ex12.cxx ex13.cxx ex14.cxx ex15.c ex16.c ex17.c ex18.c
EXAMPLESF       = ex3f.F ex4f.F ex4f90.F90 ex5f90.F90 ex8f90.F90 ex10f90.F90
MANSEC          = Sys
CLEANFILES      = binaryoutput binaryoutput.info

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex3f: ex3f.o   chkopts
	-${FLINKER} -o ex3f ex3f.o  ${PETSC_SYS_LIB}
	${RM} -f ex3f.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex4f: ex4f.o   chkopts
	-${FLINKER} -o ex4f ex4f.o  ${PETSC_SYS_LIB}
	${RM} -f ex4f.o

ex4f90: ex4f90.o   chkopts
	-${FLINKER} -o ex4f90 ex4f90.o  ${PETSC_SYS_LIB}
	${RM} -f ex4f90.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_SYS_LIB}
	${RM} -f ex5.o

ex5f90: ex5f90.o  chkopts
	-${FLINKER} -o ex5f90 ex5f90.o ${PETSC_SYS_LIB}
	${RM} ex5f90.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_SYS_LIB}
	${RM} -f ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_SYS_LIB}
	${RM} -f ex7.o

ex8f90: ex8f90.o  chkopts
	-${FLINKER} -o ex8f90 ex8f90.o ${PETSC_SYS_LIB}
	${RM} ex8f90.o

ex9: ex9.o   chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_SYS_LIB}
	${RM} -f ex9.o

ex10f90: ex10f90.o   chkopts
	-${FLINKER} -o ex10f90 ex10f90.o  ${PETSC_SYS_LIB}
	${RM} -f ex10f90.o

ex11: ex11.o   chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_SYS_LIB}
	${RM} -f ex11.o

ex12: ex12.o   chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_SYS_LIB}
	${RM} -f ex12.o

ex13: ex13.o   chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_SYS_LIB}
	${RM} -f ex13.o

ex14: ex14.o   chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_SYS_LIB}
	${RM} -f ex14.o

ex15: ex15.o   chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_SYS_LIB}
	${RM} -f ex15.o

ex16: ex16.o   chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_SYS_LIB}
	${RM} -f ex16.o

ex17: ex17.o   chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_SYS_LIB}
	${RM} -f ex17.o

ex18: ex18.o   chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_SYS_LIB}
	${RM} -f ex18.o

include ${PETSC_DIR}/lib/petsc/conf/test
