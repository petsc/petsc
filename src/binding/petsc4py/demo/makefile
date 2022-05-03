-include ../../../../petsckdir.mk
.PHONY:all
all: petsc
	${MAKE} -C binary-io
	${MAKE} -C kspsolve
	${MAKE} -C bratu2d
	${MAKE} -C bratu3d
	${MAKE} -C poisson2d
	${MAKE} -C poisson3d
	${MAKE} -C perftest
	${MAKE} -C wrap-swig
	${MAKE} -C wrap-f2py

.PHONY:petsc
petsc:
	${MAKE} -C petsc-examples
