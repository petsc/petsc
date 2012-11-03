#
#  All the Minpack coloring routines have been converted to C with the F2C 
# converter. We do not include in the distribution the original Fortran 77 
# code in the interest of saving disk space. 
#
#  All of the MINPACK routine names have been prefixed with MINPACK
# to prevent name-space pollution.
#
#  Original Fortran 77 version for Minpack-1
#   - August 1984. Thomas F. Coleman, Burton S. Garbow, Jorge J. More'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = degr.c setr.c dsm.c ido.c numsrt.c seq.c slo.c color.c \
           scolor.c
SOURCEH  = color.h
LIBBASE  = libpetscmat
LOCDIR   = src/mat/color/
MANSEC   = MatOrderings

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
