# $Id: makefile,v 1.5 1999/07/17 16:54:04 balay Exp bsmith $ 

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
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

SOURCEC        = inpututils.c
SOURCEF        =
SOURCEH        = 
OBJSC          = inpututils.o
DOCS           = readme
CLEANFILES     = 
EXAMPLESC      = input.c triconvert.c
TESTEXAMPLES_1 = input.PETSc triconvert.PETSc

input: chkopts input.o
	-${CLINKER} -o input input.o ${PETSC_LIB}
	-${RM} input.o

triconvert: chkopts triconvert.o
	-${CLINKER} -o triconvert triconvert.o ${PETSC_LIB}
	-${RM} triconvert.o

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




