# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR     = src/contrib/oberman/laplacian_q1/scalable/

EXAMPLESC      = main.c appload.c appalgebra.c appelement.c apppartition.c
EXAMPLESO      = main.o appload.o appalgebra.o appelement.o apppartition.o
EXAMPLESCH      = appctx.h

main: chkopts ${EXAMPLESO}
	-$(RM) main
	-$(CLINKER) -o main ${EXAMPLESO} $(PETSC_LIB)
	-$(RM) ${EXAMPLESO}

include $(PETSC_DIR)/bmake/common/base
include $(PETSC_DIR)/bmake/common/test
