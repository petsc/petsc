PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      =	  $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	      = 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LOCDIR        = src/viewer/examples/
LIBBASE	      = libpetscvec
EXAMPLES_2    = 
EXAMPLESC     = ex1.c ex2.c ex1.m
RUNEXAMPLES_2 = 
EXAMPLES_4    = ex1 ex2
RUNEXAMPLES_4 = 

runex2:
	@echo "There are no externally runable examples in this directory"

ex1: ex1.o   chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o   chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

