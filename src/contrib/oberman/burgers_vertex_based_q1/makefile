# $Id: makefile,v 1.3 1999/11/10 03:22:28 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR     = src/usg/2d/constant_bilinear/

TESTEXAMPLES_1 = main.PETSc main.rm
EXAMPLESC      = main.c appview.c appload.c functions.c integrator.c

main: chkopts main.o appview.o appload.o functions.o integrator.o
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o functions.o $(PETSC_LIB)

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

