# $Id: makefile,v 1.8 2000/08/01 20:58:18 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR         = src/contrib/oberman/laplacian_q1
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

TESTEXAMPLES_1 = main.PETSc main.rm

include $(PETSC_DIR)/bmake/common_test


