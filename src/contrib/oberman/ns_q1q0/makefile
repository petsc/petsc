# $Id: makefile,v 1.2 1999/11/24 21:55:48 bsmith Exp balay $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------

ALL: main

SOURCEC    = 
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 
LOCDIR     = src/usg/2d/constant_bilinear/

main: chkopts main.o appview.o appload.o functions.o integrator.o appcreate.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o appcreate.o functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o integrator.o appcreate.o


include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
include $(PETSC_DIR)/bmake/common_test




