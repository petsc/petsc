# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS	 = $(CPPFLAGS) -D__SDIR__='"$(LOCDIR)"' -I../../src
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
LOCDIR   = src/contrib/sda/examples/tutorials/

EXAMPLES_2    =
EXAMPLES_3    =

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

PETSC_CONTRIB = -L$(LDIR) -lpetsccontrib

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o $(PETSC_CONTRIB)  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(FLINKER) -o ex2 ex2.o  $(PETSC_FORTRAN_LIB) $(PETSC_CONTRIB) \
                    $(SLES_LIB)
	$(RM) -f ex2.o

