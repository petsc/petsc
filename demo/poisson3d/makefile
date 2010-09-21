# -*- makefile -*-

MPIEXEC=
PYTHON=python

.PHONY:test
test: run clean

.PHONY:run
run: run_py run_cc

.PHONY:run_py
run_py: poisson3d.so
	${MPIEXEC} ${PYTHON} poisson3d.py -ksp_monitor

.PHONY:run_cc
run_cc: poisson3d.exe
	${MPIEXEC} ./poisson3d.exe -ksp_monitor

poisson3d.so:
	${MAKE} -f makefile.f2py
poisson3d.exe:
	${MAKE} -f makefile.petsc

clean:
	${RM} *.py[co] *.o *.so *.exe 
