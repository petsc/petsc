# $Id: makefile,v 1.1 1998/04/01 23:08:29 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = $(CPPFLAGS)  -D__SDIR__='"$(LOCDIR)"'
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Note: "make clean" removes the files
#    *.o 
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

ALL: main

SOURCEC    = 
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 
LOCDIR     = src/dm/usg/2d/constant_bilinear/

main: chkopts main.o appview.o appload.o
	-$(CLINKER) -o main main.o appview.o appload.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




