
# -----------------------------------------------------------------------
# Specify user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR     = src/contrib/oberman/burgers_vertex_based_q1/

EXAMPLESC      = main.c appview.c appload.c functions.c integrator.c
EXAMPLESCH     = appctx.h

main: chkopts main.o appview.o appload.o functions.o integrator.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o functions.o $(PETSC_SNES_LIB)

include $(PETSC_DIR)/bmake/common/base

TESTEXAMPLES_1 = main.PETSc main.rm
include $(PETSC_DIR)/bmake/common/test
