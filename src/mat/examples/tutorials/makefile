# $Id: makefile,v 1.6 1997/09/10 18:47:00 balay Exp balay $ 

ALL: ex1

CFLAGS 	        = $(CPPFLAGS) $(FCONF) -D__SDIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscmat
LOCDIR          = src/mat/examples/tutorials/
TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_6  =
TESTEXAMPLES_7  =
TESTEXAMPLES_9  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_3 =
BUILDEXAMPLES_4 =
BUILDEXAMPLES_6 =
BUILDEXAMPLES_7 =
BUILDEXAMPLES_9 =

EXAMPLESC       = ex1.c
EXAMPLESF       =

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

ex1: ex1.o  chkopts
	-$(CLINKER)  -o ex1 ex1.o $(PETSC_LIB) 
	$(RM) ex1.o



