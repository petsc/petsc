ITOOLSDIR = ../..
ALL: ex1

CFLAGS   = $(OPT) -I$(ITOOLSDIR) $(COPT) $(CONF) -I$(ITOOLSDIR)/include
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1
EXAMPLESC= ex1.c
SYSLIB   = $(LDIR)/cmain$(COMM).o $(LDIR)/libpetscvec.a \
           $(LDIR)/libpetscsys.a $(BLAS) $(CLIB) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(SYSLIB)
	$(RM) -f ex1.o

include $(ITOOLSDIR)/bmake/$(PARCH)


