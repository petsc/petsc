
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
LOCDIR     = src/contrib/oberman/ns_q2q1/

main: chkopts main.o appload.o appcreate.o appsetref.o integrator.o functions.o appview.o 
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o appcreate.o appsetref.o \
          functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o integrator.o appcreate.o  appsetref.o 

include $(PETSC_DIR)/bmake/common/base
include $(PETSC_DIR)/bmake/common/test

biquadinput: chkopts biquadinput.o
	-${CLINKER} -o biquadinput biquadinput.o ${PETSC_LIB}
	-$(RM) biquadinput.o



