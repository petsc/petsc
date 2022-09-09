-include ../../../petscdir.mk

LOCDIR          = src/sys/tests/
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

testex55:
	-@export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}${PETSC_MPI4PY_PYTHONPATH}; \
         echo "Using PYTHONPATH=$${PYTHONPATH} to locate mpi4py for test"; \
         ${MPIEXEC} -n 2 ${PYTHON_EXE} ./ex55.py > ex55_1.tmp  2>&1;\
	 if (${DIFF} output/ex55_1.testout ex55_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ mpi4py python example src/sys/tests/ex55.py run successfully with 2 MPI process"; \
         else \
           echo "Possible error running Python src/sys/tests/ex55.py with 2 MPI ranks"; \
           echo "See https://petsc.org/release/faq/";\
           cat ex55_1.tmp; \
           touch ${PETSC_DIR}/check_error;\
         fi; \
         ${RM} -f ex55.tmp
