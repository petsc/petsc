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
TESTEXAMPLES_1  = ex1.PETSc ex1.rm
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = 
TESTEXAMPLES_4  =  
BUILDEXAMPLES_1 =
BUILDEXAMPLES_3 =

EXAMPLESC       = ex1.c
EXAMPLESF       = 
#

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex1f: ex1f.o  chkopts
	-$(CLINKER) -o ex1f ex1f.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex1f.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
