# $Id: makefile,v 1.7 2000/05/06 19:58:56 bsmith Exp bsmith $ 

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
EXAMPLESC      = main.c appview.c appload.c appalgebra.c appelement.c
EXAMPLESO      = main.o appview.o appload.o appalgebra.o appelement.o
EXAMPLECH      = appctx.h
DOCS           = finput.c

main: chkopts ${EXAMPLESO}
	-$(RM) main
	-$(CLINKER) -o main ${EXAMPLESO} $(PETSC_LIB)
	-$(RM) ${EXAMPLESO}

finput: chkopts finput.o
	-$(RM) finput
	-$(CLINKER) -o finput finput.o $(PETSC_LIB)
	-$(RM) finput.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




