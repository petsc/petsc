-include ../../../../petscdir.mk

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

V ?= 0
QUIET = $(shell [ "$(V)" = "0" ] && echo @)

ex7: ex7f.o
ex21f90: ex21.o
ex43f: ex43.o

testex31: ex31.PETSc
	-${QUIET}${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex31 > ex31.tmp 2>&1;
	-${QUIET}${DIFF} output/ex31_1.testout ex31.tmp > ex31.dtmp;
	-@if [ ! -s ex31.dtmp ];  then \
	  echo "C/C++ example src/vec/vec/tutorials/ex31 run successfully with MATLAB engine"; \
	else \
		echo "Possible error running C/C++ src/vec/vec/tutorials/ex31 with MATLAB engine"; \
		echo "See https://petsc.org/release/faq/";\
		touch ../../../../check_error;\
		cat ex31.tmp;\
	fi;\
	${RM} -f ex31.tmp ex31.dtmp;
