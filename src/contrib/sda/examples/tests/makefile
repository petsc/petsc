# $Id: makefile,v 1.15 1999/01/12 23:18:07 bsmith Exp balay $ 

ALL: lib

CFLAGS	 =
FFLAGS   =
CPPFLAGS =
FPPFLAGS =
LOCDIR   = src/contrib/sda/examples/tests/

EXAMPLES_2    = ex1
EXAMPLES_3    = ex2
EXAMPLESC     = ex1.c
EXAMPLESF     = ex2.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_CONTRIB}  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${FLINKER} -o ex2 ex2.o  ${PETSC_FORTRAN_LIB} ${PETSC_CONTRIB} \
                    ${PETSC_LIB}
	${RM} -f ex2.o

#---------------------------------------------------------------------------------

runex1:
	-@for d in -1d -2d -3d; do \
            for np in 1 2 3 4; do \
              for per in 0 1; do \
                for dof in 1 2 3; do \
                  for stencil_width in 1 2; do \
                    for stencil_type in 0 1; do \
                      echo $$np $$d $$per $$dof $$stencil_width $$stencil_type; \
                      ${MPIRUN} -np $$np ex1 $$d -periodic $$per -dof $$dof -stencil_width $$stencil_width -stencil_type $$stencil_type ;\
                    done; \
                  done; \
                done; \
              done; \
            done; \
          done;

include ${PETSC_DIR}/bmake/common_test
