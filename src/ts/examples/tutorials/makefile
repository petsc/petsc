#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"' 
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscts
LOCDIR          = src/ts/examples/tutorials/
TESTEXAMPLES_1  =
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = 
TESTEXAMPLES_4  =  
BUILDEXAMPLES_1 =

EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = 
#

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o -L$(LDIR) -lpetscts $(PETSC_LIB)
	$(RM) ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o -L$(LDIR) -lpetscts $(PETSC_LIB)
	$(RM) ex2.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
