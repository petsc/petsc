# $Id: makefile,v 1.2 1998/06/12 22:12:29 bsmith Exp balay $ 

ALL: input

# -----------------------------------------------------------------------
# Additional compiler/preprocessor flags
# CFLAGS/FFLAGS    -  Additional compiler flags if required by the application
# CPPFLAGS/FPPFLAGS - Additional preprocessor flags for c/fortran
# -----------------------------------------------------------------------
CFLAGS    = 
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LOCDIR     = src/contrib/flash/
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Note: "make clean" removes the files
#    *.o 
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

SOURCEC    = input.c
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 

input: chkopts input.o
	-${CLINKER} -o input input.o ${PETSC_LIB}

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




