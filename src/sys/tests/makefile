-include ../../../petscdir.mk

LOCDIR          = src/sys/tests/
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

testex55:
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/sys/tests/ex55.py with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/sys/tests/ex55.py manually"; \
	else \
           export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}${PETSC_MPI4PY_PYTHONPATH}; \
           echo "Using PYTHONPATH=$${PYTHONPATH} to locate mpi4py for test"; \
           ${MPIEXEC} -n 2 ${PYTHON_EXE} ./ex55.py > ex55_1.tmp  2>&1;\
	   if (${DIFF} output/ex55_1.testout ex55_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ mpi4py python example src/sys/tests/ex55.py run successfully with 2 MPI process"; \
	   else echo "Possible error running Python src/sys/tests/ex55.py with 2 MPI processes"; \
           echo "See https://petsc.org/release/faq/";\
           cat ex55_1.tmp; fi; \
        ${RM} -f ex55.tmp; fi
