# $Id: makefile,v 1.15 1999/02/02 21:27:03 balay Exp bsmith $ 
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
LOCDIR   = src/mat/impls/color/
MANSEC   = Mat

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
