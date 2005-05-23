# $Id: makefile,v 1.1 1998/10/23 19:23:48 oberman Exp $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = $(CPPFLAGS)  -D__SDIR__='"$(LOCDIR)"'
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Note: "make clean" removes the files
#    *.o 
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

ALL: main

SOURCEC    = 
SOURCEF    =
SOURCEH    = 
OBJSC      = 
CLEANFILES = 
LOCDIR     = src/contrib/src/ns_q1q0_penalty/

main: chkopts main.o appview.o appload.o functions.o integrator.o appcreate.o
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o appcreate.o functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o integrator.o appcreate.o

penaltyinput: chkopts penaltyinput.o
	-${CLINKER} -o penaltyinput penaltyinput.o ${PETSC_LIB}
	-$(RM) penaltyinput.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




