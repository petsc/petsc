IPETSCDIR = ../../..
ALL: ex1

CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF) 
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   =
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscvec
EXAMPLES  = ex2
REXAMPLES = runex2
CEXAMPLES = ex1 ex2
EXAMPLESC = ex1.c ex2.c
LINCLUDE  = $(SOURCEH)

runex2:
	-@$(MPIRUN)  ex2  > ex2_1.tmp 2>&1;\
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi ;\
           $(RM) -f ex2_1.tmp

ex1: ex1.o chkopts
	-$(CLINKER) -o ex1 ex1.o  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o  $(SLES_LIB)
	$(RM) -f ex2.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
