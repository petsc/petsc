# $Id: base.g,v 1.11 1997/09/10 18:10:05 balay Exp bsmith $ 

PETSCFLAGS = -DUSE_PETSC_DEBUG  -DUSE_PETSC_LOG -DUSE_PETSC_BOPT_g \
             -DUSE_PETSC_STACK

COPTFLAGS  = -g -Wall -Wshadow
#
# To prohibit Fortran implicit typing, add -u in FOPTFLAGS definition
#
#FOPTFLAGS  = -g -dalign
FOPTFLAGS  = -g -dalign
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

