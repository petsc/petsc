IPETSCDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF) -I$(IPETSCDIR)/include
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
CEXAMPLES= ex1 ex2
EXAMPLESC= ex1.c ex2.c
SYSLIB   = -L$(LDIR) -lpetscstencil -lpetscgrid -lpetscmat -lpetscvec \
           -lpetscdraw -lpetscsys $(BS_LIB) $(LAPACK_LIB) \
           $(BLAS_LIB) $(FC_LIB) $(X11_LIB) $(MPI_LIB) $(SYS_LIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	@echo "There are no externally runable examples in this directory"
	
ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(SYSLIB)
	$(RM) -f ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o  $(SYSLIB)
	$(RM) -f ex2.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

