ITOOLSDIR = ../..
ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR) $(COPT) -I$(ITOOLSDIR)/include
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
SYSLIB   = $(LDIR)/cmain$(COMM).o $(LDIR)/$(COMPLEX)/libpetscvec.a \
           $(LDIR)/libpetscvec.a \
           $(LDIR)/libpetscsystem.a $(BLAS) $(CLIB) $(SLIB) -lm 
LOCAL_INCLUDE = $(SOURCEH)

ex1: ex1.o 
	$(CLINKER) -o ex1 ex1.o  $(SYSLIB)
	$(RM) ex1.o


