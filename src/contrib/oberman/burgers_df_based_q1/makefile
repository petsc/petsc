# $Id: makefile,v 1.1 1999/11/15 16:04:32 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#    
TESTEXAMPLES_1 = main.PETSc main.rm
EXAMPLESC      = main.c appview.c appload.c functions.c integrator.c appctx.h appcreate.c

ALL: main

LOCDIR     = src/usg/2d/constant_bilinear/

main: chkopts main.o appview.o appload.o functions.o integrator.o appcreate.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o appcreate.o functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o integrator.o appcreate.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




