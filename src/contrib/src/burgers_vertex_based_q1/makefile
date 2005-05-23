# $Id: makefile,v 1.1 1998/10/23 19:26:24 oberman Exp $ 

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
LOCDIR     = src/contrib/src/burgers_vertex_based_q1/

main: chkopts main.o appview.o appload.o functions.o integrator.o
	-$(CLINKER) -o main main.o appview.o appload.o integrator.o functions.o $(PETSC_LIB)
	-$(RM) main.o appview.o appload.o functions.o integrator.o

functions.o:
	cc -c functions.c

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

integrator.o: chkopts appctx.h
	-$(CLINKER) -c integrator.c