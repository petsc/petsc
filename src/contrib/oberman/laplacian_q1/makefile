# $Id: makefile,v 1.1 1999/11/15 16:02:24 bsmith Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
#    

ALL: main

LOCDIR     = src/usg/2d/constant_bilinear/

TESTEXAMPLES_1 = main.PETSc main.rm
EXAMPLESC      = main.c appview.c appload.c functions.c appsetalg.c

main: chkopts main.o appview.o appload.o functions.o  appsetalg.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o functions.o appsetalg.o $(PETSC_LIB)
#	-$(RM) main.o appview.o appload.o functions.o appcreate.o appintegrator.o appsetref.o appsetalg.o


include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




