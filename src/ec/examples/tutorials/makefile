# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__SDIR__='"$(LOCDIR)"'
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

