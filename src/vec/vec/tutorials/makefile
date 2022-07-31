-include ../../../../petscdir.mk

MANSEC           = Vec
LOCDIR		 = src/vec/vec/tutorials/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex7: ex7.o ex7f.o
ex21f90: ex21f90.o ex21.o
ex43f: ex43.o ex43f.o

testex31: ex31.PETSc
	-@${MPIEXEC} -n 1 ./ex31 > ex31_1.tmp 2>&1; \
	 if (${DIFF} output/ex31_1.testout ex31_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/vec/vec/tutorials/ex31 run successfully with Matlab engine"; \
	 else \
           echo "Possible error running C/C++ src/vec/vec/tutorials/ex31 with Matlab engine"; \
           echo "See https://petsc.org/release/faq/";\
           touch ${PETSC_DIR}/check_error;\
           cat ex31_1.tmp;\
         fi

