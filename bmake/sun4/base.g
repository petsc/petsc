PETSCFLAGS = -DPETSC_DEBUG  -DPETSC_LOG -DPETSC_BOPT_g -Dlint
COPTFLAGS  = -g -Wall
#
# To use Fortran implicit typing, remove -u in BASEOPTF definition
#
FOPTFLAGS  = -g -dalign
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

