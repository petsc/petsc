# $Id: makefile,v 1.1 1998/10/23 19:11:15 oberman Exp $ 

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
LOCDIR     = src/contrib/oberman/quasigeo_q1/
FILES = main.o appload.o appcreate.o appsetref.o appsetalg.o appintegrator.o functions.o appview.o 

main: chkopts $(FILES)
	-$(CLINKER) -o main $(FIILES) $(PETSC_LIB)
	-$(RM) $(FILES)

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

biquadinput: chkopts biquadinput.o
	-${CLINKER} -o biquadinput biquadinput.o ${PETSC_LIB}
	-$(RM) biquadinput.o
