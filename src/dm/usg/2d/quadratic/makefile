# $Id: makefile,v 1.2 1998/04/28 00:58:25 bsmith Exp balay $ 

# -----------------------------------------------------------------------
# Specify user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
FFLAGS    =
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
LOCDIR     = src/dm/usg/2d/quadratic/

main: chkopts main.o appview.o appload.o
	-$(CLINKER) -o main main.o appview.o appload.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




