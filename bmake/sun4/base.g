# $Id: base.g,v 1.12 1997/10/19 03:20:08 bsmith Exp bsmith $ 

PETSCFLAGS = -DUSE_PETSC_DEBUG  -DUSE_PETSC_LOG -DUSE_PETSC_BOPT_g \
             -DUSE_PETSC_STACK -DUSE_DYNAMIC_LIBRARIES

COPTFLAGS  = -g -Wall -Wshadow
#
# To prohibit Fortran implicit typing, add -u in FOPTFLAGS definition
#
#FOPTFLAGS  = -g -dalign
FOPTFLAGS  = -g -dalign
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

