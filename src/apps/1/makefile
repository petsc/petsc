# $Id: makefile,v 1.1 1997/11/04 23:49:31 bsmith Exp bsmith $ 

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
LOCDIR     = src/apps/1/

ex1: chkopts ex1.o
	-$(CLINKER) -o ex1 ex1.o $(PETSC_LIB)
	-$(RM) ex1.o

ex2: chkopts ex2.o appview.o appload.o
	-$(CLINKER) -o ex2 ex2.o appview.o appload.o $(PETSC_LIB)
	-$(RM) ex2.o appview.o appload.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




