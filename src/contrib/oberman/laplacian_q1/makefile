# $Id: makefile,v 1.4 2000/01/16 03:28:41 bsmith Exp bsmith $ 

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
EXAMPLESC      = main.c appview.c appload.c functions.c appalgebra.c appelement.c
EXAMPLESO      = main.o appview.o appload.o functions.o appalgebra.o appelement.o

main: chkopts ${EXAMPLESO}
	-$(RM) main
	-$(CLINKER) -o main ${EXAMPLESO} $(PETSC_LIB)
	-$(RM) ${EXAMPLESO}

finput: chkopts finput.o
	-$(RM) finput
	-$(CLINKER) -o finput finput.o $(PETSC_LIB)
	-$(RM) finput.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




