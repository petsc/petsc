IPETSCDIR = ../../../
ALL: ex1

CFLAGS    = $(OPT) $(CONF) $(INCLUDEDIRS)
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   =
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscsles
EXAMPLES  = ex1
EXAMPLESC = ex1.c
SYSLIB    = -L$(LDIR) -lpetscsles -lpetscksp -lpetscmat \
           -lpetscvec -lpetscsys -lpetscdraw \
           $(LAPACK_LIB) $(BLAS_LIB) $(X11_LIB) $(MPI_LIB) $(SYS_LIB) \
           $(FC_LIB) -lm 
LINCLUDE  = $(SOURCEH)

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(LIBNAME) $(SYSLIB)
	$(RM) ex1.o


include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
#
#   Makefile for Yale sparse matrix package
#

OBJ = cdrv.o md.o mdi.o mdm.o mdp.o mdu.o ndrv.o nnf.o \
      nnfc.o nns.o nnsc.o nnt.o nntc.o nroc.o nsf.o \
      nsfc.o odrv.o sdrv.o snf.o sns.o sro.o ssf.o \
      tdrv.o trk.o r1mach.o xsetf.o xsetun.o d1mach.o \
      i1mach.o xermax.o xerror.o xerrwv.o j4save.o  \
      xerprt.o xerabt.o fdump.o xerctl.o xgetua.o xersav.o

test1:  ysmp.o $(PETSC_ARCH)ysmp.a
	f77 -o test1 ysmp.o $(PETSC_ARCH)ysmp.a
	rm -f ysmp.o

$(PETSC_ARCH)ysmp.a  : $(OBJ)
	ar r $(PETSC_ARCH)ysmp.a $?
	ranlib $(PETSC_ARCH)ysmp.a
	rm -f  $(OBJ)
