-include ../../../../petscdir.mk

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

V ?= 0
QUIET = $(shell [ "$(V)" = "0" ] && echo @)

runex47: ex47.PETSc
	-${QUIET}H5OUT=`mktemp -t petsc.h5.XXXXXX`; ${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex47 -filename $${H5OUT} > ex47.tmp 2>&1;
	-@if [ ! -s ex47.tmp ]; then \
		echo "C/C++ example src/vec/vec/tests/ex47 run successfully with HDF5"; \
	else \
		printf "${PWD}\nPossible problem with ex47 running with HDF5, diffs above\n=========================================\n";\
		touch ../../../../check_error;\
	fi; \
	${RM} -f ex47.tmp $${H5OUT}
