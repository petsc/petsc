IPETSCDIR = ../../..
ALL: ex1

CFLAGS        = $(OPT) $(INCLUDEDIRS) $(CONF) -I$(IPETSCDIR)/include
SOURCEC       = 
SOURCEF       =
WSOURCE       = 
SOURCEH       =
OBJSC         =
WOBJS         = 
OBJSF         =
LIBBASE       = libpetscvec
EXAMPLES_2    = ex2
EXAMPLESC     = ex2.c
RUNEXAMPLES_2 = runex2

runex2:
	@echo "There are no externally runable examples in this directory"
	
ex2: ex2.o   chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

