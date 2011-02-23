# -*- makefile -*-

MPIEXEC=
PYTHON=python

.PHONY:test
test: run clean

.PHONY:run
run: run_py run_cc

SCRIPT=driver
MODULE=App
.PHONY:run_py
run_py: ${MODULE}.so
	${MPIEXEC} ${PYTHON} ${SCRIPT}.py

EXECUTABLE=driver
.PHONY:run_cc
run_cc: ${EXECUTABLE}.exe
	${MPIEXEC} ./${EXECUTABLE}.exe

${MODULE}.so:
	${MAKE} -f makefile.f2py

${EXECUTABLE}.exe:
	${MAKE} -f makefile.petsc \
        PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH}

.PHONY:clean
clean:
	${RM} -r __pycache__ *.py[co]
	${RM} ${MODULE}.so ${EXECUTABLE}.exe
