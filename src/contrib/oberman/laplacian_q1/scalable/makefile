# $Id: makefile,v 1.3 2001/03/22 20:33:03 bsmith Exp balay $ 

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

main.$(BOPT): chkopts ${EXAMPLESO}
	-$(RM) main.$(BOPT)
	-$(CLINKER) -o main.$(BOPT) ${EXAMPLESO} $(PETSC_LIB)
	-$(RM) ${EXAMPLESO}

include $(PETSC_DIR)/bmake/common/base
include $(PETSC_DIR)/bmake/common/test
