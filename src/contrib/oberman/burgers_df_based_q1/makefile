
# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#    
EXAMPLESC      = main.c appview.c appload.c functions.c appctx.h appsetalg.c
EXAMPLESCH     = appctx.h

ALL: main

LOCDIR     = src/contrib/oberman/burgers_df_based_q1/

main: chkopts main.o appview.o appload.o functions.o appsetalg.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o appsetalg.o functions.o $(PETSC_SNES_LIB)
	-$(RM) main.o appview.o appload.o functions.o appsetalg.o

include $(PETSC_DIR)/bmake/common/base

TESTEXAMPLES_1 = main.PETSc main.rm

include $(PETSC_DIR)/bmake/common/test


