# $Id: makefile,v 1.5 2000/08/17 20:02:26 balay Exp balay $ 

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

EXAMPLESC      = main.c appview.c appload.c functions.c integrator.c
EXAMPLESCH     = appctx.h

main: chkopts main.o appview.o appload.o functions.o integrator.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o functions.o $(PETSC_LIB)

include $(PETSC_DIR)/bmake/common/base

TESTEXAMPLES_1 = main.PETSc main.rm
include $(PETSC_DIR)/bmake/common/test
