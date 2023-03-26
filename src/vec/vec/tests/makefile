-include ../../../../petscdir.mk

MANSEC          = Vec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

runex47: ex47.PETSc
	-@H5OUT=`mktemp -t petsc.h5.XXXXXX`; ${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex47 -filename $${H5OUT} > ex47_1.tmp 2>&1; \
	   if (${DIFF} output/ex47_1.out ex47_1.tmp) then \
             echo "C/C++ example src/vec/vec/tests/ex47 run successfully with hdf5"; \
           else \
             printf "${PWD}\nPossible problem with ex47 running with hdf5, diffs above\n=========================================\n";\
             touch ${PETSC_DIR}/check_error;\
           fi; \
	   ${RM} -f ex47_1.tmp $${H5OUT}
