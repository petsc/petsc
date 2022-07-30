-include ../../../../petscdir.mk

LOCDIR          = src/vec/vec/tests/
MANSEC          = Vec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
-include ../petscdir.mk

runex47: ex47.PETSc
	-@H5OUT=`mktemp -t petsc.h5.XXXXXX`; ${MPIEXEC} -n 1 ./ex47 -filename $${H5OUT} > ex47_1.tmp 2>&1; \
	   if (${DIFF} output/ex47_1.out ex47_1.tmp) then \
           echo "C/C++ example src/vec/vec/tests/ex47 run successfully with hdf5"; \
           else  printf "${PWD}\nPossible problem with ex47 running with hdf5, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex47_1.tmp $${H5OUT}
