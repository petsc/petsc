# $Id: makefile,v 1.9 1998/03/19 02:08:25 balay Exp balay $ 
#
#  All the Minpack coloring routines have been converted to C with the F2C 
# converter. We do not include in the distribution the original Fortran 77 
# code in the interest of saving disk space. 
#
ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = degr.c setr.c dsm.c ido.c numsrt.c seq.c slo.c color.c \
           scolor.c
SOURCEH  = color.h
OBJSC    = degr.o setr.o dsm.o ido.o numsrt.o seq.o slo.o color.o \
           scolor.o
DOCS     = readme
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/color/
MANSEC   = 2

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
