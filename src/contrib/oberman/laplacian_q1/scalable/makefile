# $Id: makefile,v 1.8 2000/08/01 20:58:18 bsmith Exp $ 

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
