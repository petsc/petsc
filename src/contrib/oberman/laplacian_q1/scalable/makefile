# $Id: makefile,v 1.1 2000/08/17 15:47:45 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main.$(BOPT)

LOCDIR     = src/contrib/oberman/laplacian_q1/

EXAMPLESC      = main.c appload.c appalgebra.c appelement.c apppartition.c
EXAMPLESO      = main.o appload.o appalgebra.o appelement.o apppartition.o
EXAMPLECH      = appctx.h
DOCS           =

main.$(BOPT): chkopts ${EXAMPLESO}
	-$(RM) main.$(BOPT)
	-$(CLINKER) -o main.$(BOPT) ${EXAMPLESO} $(PETSC_LIB)
	-$(RM) ${EXAMPLESO}

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
include $(PETSC_DIR)/bmake/common_test
