# $Id: base.g,v 1.14 1997/12/20 15:28:58 bsmith Exp bsmith $ 

PETSCFLAGS = -DUSE_PETSC_DEBUG  -DUSE_PETSC_LOG -DUSE_PETSC_BOPT_g \
             -DUSE_PETSC_STACK

#
#  Add -DUSE_DYNAMIC_LIBRARIES to use dynamic loading of KSP routines,
# MUST use shared versions of library to work
#

COPTFLAGS  = -g -Wall -Wshadow
#
# To prohibit Fortran implicit typing, add -u in FOPTFLAGS definition
#
#FOPTFLAGS  = -g -dalign
FOPTFLAGS  = -g -dalign
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

