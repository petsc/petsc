# $Id: makefile,v 1.18 1999/11/05 14:46:11 bsmith Exp bsmith $ 
#
#  All the Minpack coloring routines have been converted to C with the F2C 
# converter. We do not include in the distribution the original Fortran 77 
# code in the interest of saving disk space. 
#
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = degr.c setr.c dsm.c ido.c numsrt.c seq.c slo.c color.c \
           scolor.c
SOURCEH  = color.h
OBJSC    = degr.o setr.o dsm.o ido.o numsrt.o seq.o slo.o color.o \
           scolor.o
DOCS     = readme
LIBBASE  = libpetscmat
LOCDIR   = src/mat/color/
MANSEC   = MatOrderings



include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
