# $Id: makefile,v 1.1 1997/12/10 22:29:13 bsmith Exp $ 

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

ALL: ex1

SOURCEC    = 
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 
LOCDIR     = src/apps/2/

ex1: chkopts ex1.o appview.o appload.o
	-$(CLINKER) -o ex1 ex1.o appview.o appload.o $(PETSC_LIB)
	-$(RM) ex1.o appview.o appload.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




