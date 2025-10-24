-include ../../../petscdir.mk

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

V ?= 0
QUIET = $(shell [ "$(V)" = "0" ] && echo @)

testex55:
	-${QUIET}export PYTHONPATH=$${PYTHONPATH:+$$PYTHONPATH:}${PETSCPYTHONPATH};\
		echo "Using PYTHONPATH=$${PYTHONPATH} to locate mpi4py for test"; \
		${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ${PYTHON_EXE} ./ex55.py > ex55.tmp  2>&1;
	-@if [ ! -s ex55.tmp ]; then \
		echo "C/C++ mpi4py python example src/sys/tests/ex55.py run successfully with 2 MPI processes"; \
	else \
		echo "Possible error running Python src/sys/tests/ex55.py with 2 MPI processes"; \
		echo "See https://petsc.org/release/faq/";\
		cat ex55.tmp; \
		touch ../../../check_error;\
	fi; \
	${RM} -f ex55.tmp
