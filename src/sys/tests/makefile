-include ../../../petscdir.mk

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

testex55:
	-@export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}${PETSCPYTHONPATH};\
         echo "Using PYTHONPATH=$${PYTHONPATH} to locate mpi4py for test"; \
         ${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ${PYTHON_EXE} ./ex55.py > ex55_1.tmp  2>&1;\
         if (test ! -s ex55_1.tmp) then \
             echo "C/C++ mpi4py Python example src/sys/tests/ex55.py run successfully with 2 MPI processes"; \
         else \
           echo "Possible error running Python example src/sys/tests/ex55.py with 2 MPI processes"; \
           echo "See https://petsc.org/release/faq/";\
           cat ex55_1.tmp; \
           touch ../../../check_error;\
         fi; \
         ${RM} -f ex55.tmp
