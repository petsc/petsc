# $Id: makefile,v 1.7 2000/01/27 16:35:23 bsmith Exp bsmith $ 

ALL: lib

# -----------------------------------------------------------------------
# Additional compiler/preprocessor flags
# CFLAGS/FFLAGS    -  Additional compiler flags if required by the application
# CPPFLAGS/FPPFLAGS - Additional preprocessor flags for c/fortran
# -----------------------------------------------------------------------
CFLAGS    = 
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LIBBASE   = libpetscdm
MANSEC    = DA
LOCDIR    = src/dm/usg/utils/
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Note: "make clean" removes the files
#    *.o 
#    

SOURCEC        = inpututils.c
SOURCEF        =
SOURCEH        = 
OBJSC          = inpututils.o
DOCS           = readme
EXAMPLESC      = input.c triconvert.c

input: chkopts input.o
	-${CLINKER} -o input input.o ${PETSC_LIB}
	-${RM} input.o

triconvert: chkopts triconvert.o
	-${CLINKER} -o triconvert triconvert.o ${PETSC_LIB}
	-${RM} triconvert.o

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

#--------------------------------------------------
TESTEXAMPLES_1 = input.PETSc input.rm triconvert.PETSc triconvert.rm

include ${PETSC_DIR}/bmake/common_test

