# $Id: makefile,v 1.184 1997/09/10 16:23:37 bsmith Exp $ 

ALL: sif

CFLAGS	        = $(CPPFLAGS) $(FCONF) -D__SDIR__='"$(LOCDIR)"' 
FFLAGS	        =
LOCDIR          = src/contrib/sif
EXAMPLESC       = 
EXAMPLESF       = sif.F

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

sif: sif.o chkopts
	-$(FLINKER) -o sif sif.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) sif.o 

