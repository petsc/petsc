# -*- makefile -*-

MPIEXEC=
PYTHON=python

F2PY = f2py
F2PY_FLAGS  = --quiet
F2PY_FLAGS += --noarch --f90flags=''
F2PY_FLAGS += -DF2PY_REPORT_ON_ARRAY_COPY=1

.PHONY:test
test: run clean

.PHONY:run
run: run_py run_cc

SCRIPT=poisson3d
MODULE=del2lib
.PHONY:run_py
run_py: ${MODULE}.so
	${MPIEXEC} ${PYTHON} ${SCRIPT}.py

${MODULE}.so: ${MODULE}.f90
	${F2PY} ${F2PY_FLAGS} -c $< -m ${MODULE}

EXECUTABLE=poisson3d
.PHONY:run_cc
run_cc: ${EXECUTABLE}.exe
	${MPIEXEC} ./${EXECUTABLE}.exe

${EXECUTABLE}.exe:
	${MAKE} -f makefile.petsc \
        PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH}

.PHONY:clean
clean:
	${RM} *.py[co] ${MODULE}*.so ${EXECUTABLE}.exe
	${RM} -r __pycache__
