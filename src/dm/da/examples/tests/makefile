IPETSCDIR = ../../..
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF) 
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1
CEXAMPLES= ex1
EXAMPLESC= ex1.c
LINCLUDE = $(SOURCEH)

runexamples:

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
