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
LIBBASE  = libpetscdraw
CEXAMPLES= ex1 ex2 ex3 ex4
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c
REXAMPLES=runex1
LINCLUDE = $(SOURCEH)

runex1:
	-@echo "Run examples in src/draw/examples manually"

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) -f ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) -f ex4.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
