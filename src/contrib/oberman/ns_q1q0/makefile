
# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------

ALL: main

SOURCEC    = 
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 
LOCDIR     = src/contrib/oberman/ns_q1q0/

main: chkopts main.o appview.o appload.o functions.o integrator.o appcreate.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o appcreate.o functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o integrator.o appcreate.o


include $(PETSC_DIR)/bmake/common/base
include $(PETSC_DIR)/bmake/common/test




