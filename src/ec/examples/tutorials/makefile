# $Id: makefile,v 1.7 1997/09/10 18:54:34 balay Exp balay $ 

ALL: ex1

CFLAGS	        = $(CPPFLAGS)  $(FCONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscec
LOCDIR          = src/ec/examples/tutorials/
TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_4  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_4 =

EXAMPLESC       = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(SLES_LIB)
	$(RM) ex1.o

