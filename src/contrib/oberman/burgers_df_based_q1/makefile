# $Id: makefile,v 1.4 1999/12/09 15:23:04 hzhang Exp bsmith $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = 
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#    
TESTEXAMPLES_1 = main.PETSc main.rm
EXAMPLESC      = main.c appview.c appload.c functions.c appctx.h appsetalg.c
EXAMPLESCH     = appctx.h

ALL: main

LOCDIR     = src/usg/2d/constant_bilinear/

main: chkopts main.o appview.o appload.o functions.o appsetalg.o
	-$(RM) main
	-$(CLINKER) -o main main.o appview.o appload.o appsetalg.o functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o appsetalg.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base




