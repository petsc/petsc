PETSCFLAGS = -DPETSC_DEBUG  -DPETSC_LOG -DPETSC_BOPT_g -Dlint
COPTFLAGS  = -g -Wall
#
# To prohibit Fortran implicit typing, add -u in FOPTFLAGS definition
#
FOPTFLAGS  = -g -dalign
SYS_LIB    = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o

