IPETSCDIR = ../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF) -D__DIR__='"sys/"' \
            -I$(IPETSCDIR)/pinclude
SOURCEC   = tr.c arch.c txt.c file.c  nreg.c try.c comsort.c destroy.c
SOURCEF   =
SOURCEH   =
OBJSC     = tr.o arch.o txt.o file.o  nreg.o try.o comsort.o destroy.o
OBJSF     =
WSOURCEC  = archf.c   
WOBJS     = archf.o  
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = error

SYSLIB   = -L$(LDIR) -lpetscstencil -lpetscgrid -lpetscmat -lpetscvec \
           $(BLAS_LIB) -lpetscdraw -lpetscsys $(LAPACK_LIB) $(BLAS_LIB) \
           $(FC_LIB) $(X11_LIB) $(MPI_LIB) $(SYS_LIB) -lm 

runexamples:

review: review.o 
	-$(CLINKER) -o review review.o  $(SYSLIB)
	$(RM) -f review.o

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
