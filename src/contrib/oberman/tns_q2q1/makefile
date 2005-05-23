# $Id: makefile,v 1.1 1999/11/22 16:13:37 bsmith Exp bsmith $ 

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
LOCDIR     = src/contrib/oberman/tns_q2q1/

main: chkopts main.o appload.o appcreate.o appsetref.o appsetalg.o integrator.o functions.o appview.o 
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o appcreate.o appsetalg.o appsetref.o functions.o $(PETSC_LIB)
#	-$(RM) main.o appview.o appload.o functions.o integrator.o appcreate.o appsetalg.o appsetref.o 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

biquadinput: chkopts biquadinput.o
	-${CLINKER} -o biquadinput biquadinput.o ${PETSC_LIB}
	-$(RM) biquadinput.o



