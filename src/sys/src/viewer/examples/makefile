IPETSCDIR = ../../..
ALL: ex1

CFLAGS        =  $(PETSC_INCLUDE) $(CONF) -I$(IPETSCDIR)/include
SOURCEC       = 
SOURCEF       =
SOURCEH       =
OBJSC         =
OBJSF         =
LIBBASE       = libpetscvec
EXAMPLES_2    = 
EXAMPLESC     = ex2.c
RUNEXAMPLES_2 = 

runex2:
	@echo "There are no externally runable examples in this directory"
	
ex2: ex2.o   chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

