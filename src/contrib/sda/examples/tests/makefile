# $Id: makefile,v 1.11 1998/03/19 02:14:41 balay Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -I../../src
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
LOCDIR   = src/contrib/sda/examples/tests/

EXAMPLES_2    = ex1
EXAMPLES_3    = ex2
EXAMPLESC     = ex1.c
EXAMPLESF     = ex2.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_CONTRIB}  ${SLES_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${FLINKER} -o ex2 ex2.o  ${PETSC_FORTRAN_LIB} ${PETSC_CONTRIB} \
                    ${SLES_LIB}
	${RM} -f ex2.o

