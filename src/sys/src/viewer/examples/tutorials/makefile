#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        =  $(CPPFLAGS) -D__SDIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LOCDIR          = src/viewer/examples/tutorials
LIBBASE	        = libpetscvec
TESTEXAMPLES_2  = 
TESTEXAMPLES_4  = 
BUILDEXAMPLES_4 = 
EXAMPLESC       = 

ex1: ex1.o   chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o   chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

