# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

PETSCFLAGS = -DPETSC_DEBUG  -DPETSC_LOG -DPETSC_BOPT_g 
COPTFLAGS  = -g -Wall -Wshadow
#
# To prohibit Fortran implicit typing, add -u in FOPTFLAGS definition
#
#FOPTFLAGS  = -g -dalign
FOPTFLAGS  = -g -dalign
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

