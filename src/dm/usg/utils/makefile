# $Id: makefile,v 1.1 1997/09/22 21:58:04 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"'
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Note: "make clean" removes the files
#    *.o 
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

ALL: input

SOURCEC    = input.c
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 
LOCDIR     = src/contrib/flash/

input: chkopts input.o
	-${CLINKER} -o input input.o ${PETSC_LIB}

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




