-include ../../../../petscdir.mk
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/vec/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
                  ex11.c ex12.c ex14.c ex15.c ex16.c ex17.c ex18.c ex21.c ex22.c \
                  ex23.c ex24.c ex25.c ex28.c ex29.c ex33.c ex34.c ex35.c \
                  ex36.c ex37.c ex38.c ex39.c ex40.c ex41.c ex42.c ex45.c ex46.c \
		  ex47.c ex49.c ex50.c ex51.c ex55.c ex56.c ex58.c
EXAMPLESCXX     = ex57.cxx
EXAMPLESF       = ex17f.F ex19f.F ex20f.F ex30f.F ex32f.F ex40f90.F90
EXAMPLESCU      = ex100.cu
MANSEC          = Vec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
-include ../petscdir.mk

runex47: ex47.PETSc
	-@${MPIEXEC} -n 1 ./ex47  > ex47_1.tmp 2>&1; \
	   if (${DIFF} output/ex47_1.out ex47_1.tmp) then \
           echo "C/C++ example src/vec/vec/tests/ex47 run successfully with hdf5"; \
           else  printf "${PWD}\nPossible problem with ex47 running with hdf5, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex47_1.tmp
