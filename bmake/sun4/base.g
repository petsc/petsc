BASEOPT = -g -Wall -DPETSC_DEBUG  -DPETSC_LOG -DPETSC_BOPT_g -Dlint
# Note:  To use Fortran implicit typing, remove -u in BASEOPTF definition
BASEOPTF = -g -dalign  -u
SYS_LIB  = /usr/lib/debug/malloc.o /usr/lib/debug/mallocmap.o


