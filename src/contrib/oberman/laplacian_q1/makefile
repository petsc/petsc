
# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR         = src/contrib/oberman/laplacian_q1/
EXAMPLESC      = main.c appview.c appload.c appalgebra.c appelement.c
EXAMPLESO      = main.o appview.o appload.o appalgebra.o appelement.o
EXAMPLESCH     = appctx.h
DOCS           = finput.c
DIRS           = scalable

main: chkopts ${EXAMPLESO}
	-$(RM) main
	-$(CLINKER) -o main ${EXAMPLESO} $(PETSC_SNES_LIB)
	-$(RM) ${EXAMPLESO}

finput: chkopts finput.o
	-$(RM) finput
	-$(CLINKER) -o finput finput.o $(PETSC_SNES_LIB)
	-$(RM) finput.o

include $(PETSC_DIR)/bmake/common/base

TESTEXAMPLES_C = main.PETSc main.rm

include $(PETSC_DIR)/bmake/common/test


