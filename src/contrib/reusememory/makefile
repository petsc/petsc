

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' ${BS_INCLUDE}
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = sda
LOCDIR   = src/contrib/reuse

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: chkopts ex1.o reuse.o
	-${CLINKER}  -o ex1 ex1.o reuse.o ${PETSC_LIB} 
	${RM} ex1.o reuse.o