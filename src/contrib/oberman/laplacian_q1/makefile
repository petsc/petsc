# $Id: makefile,v 1.3 1999/11/23 18:37:03 hzhang Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR     = src/contrib/oberman/laplacian_q1

TESTEXAMPLES_1 = main.PETSc main.rm
EXAMPLESC      = main.c appview.c appload.c functions.c appalgebra.c

main: chkopts main.o appview.o appload.o functions.o  appalgebra.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o functions.o appalgebra.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o appcreate.o appintegrator.o appsetref.o appalgebra.o


include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




