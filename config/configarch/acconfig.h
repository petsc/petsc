/* The PRIMARY source of this file is acconfig.h */
/* These are needed for ANY declaration that may be made by an AC_DEFINE */

/* Define if 64 bit values */
#undef HAVE_64BITS

/* Define if Fortran external names are all caps */
#undef HAVE_FORTRAN_CAPS

/* Define if Fortran external names have one trailing underscore */
#undef HAVE_FORTRAN_UNDERSCORE

/* Define if Fortran external names have two trailing underscores */
#undef HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

/* Define if Fortran external names have no trailing underscore */
#undef HAVE_FORTRAN_NOUNDERSCORE

/* Define if getdomainname function available */
#undef HAVE_GETDOMAINNAME

/* Define if bytes should be swapped in generic binary output */
#undef HAVE_SWAPPED_BYTES

/* Define if stdarg can be used */
#undef USE_STDARG

/* Define if doubles must be double aligned */
#undef HAVE_DOUBLES_ALIGNED

/* Define if X11 is available */
#undef HAVE_X11

/* Define as the string that has the name of the architecture */
#undef PETSC_ARCH_NAME

/* Define if the SUNDIALS package is available */
#undef HAVE_SUNDIALS

/* Define in the MPICH MPE library is available */
#undef HAVE_MPE

/* Define if MPI_Request_free does NOT work (some IBM systems) */
#undef HAVE_BROKEN_REQUEST_FREE

/* If using the Fortran BLAS, define this */
#undef HAVE_SLOW_NRM2

/* PETSC Arch name */
#undef ARCH_NAME

/* /usr/ucb/ps */
#undef HAVE_UCBPS

/* Define if SIGSYS is not defined */
#undef MISSING_SIGSYS

/* Define if SIGBUS is not defined */
#undef MISSING_SIGBUS

/* Define if SIGQUIT is not defined */
#undef MISSING_SIGQUIT

/* Define if v.printf requires the last argument to be cast as char * */
#undef HAVE_VPRINTF_CHAR

/* Define if free returns an int */
#undef HAVE_FREE_RETURN_INT

/* Define if GETPWUID is not available */
#undef MISSING_GETPWUID

/* Define is socket is not available */
#undef MISSING_SOCKETS

/* Define if Fortran does not allow integer * 4 etc. */
#undef MISSING_FORTRANSTAR

/* Define if DREAL is missing */
#undef MISSING_DREAL

/* Define if CPP macros cannot be recursive */
#undef HAVE_BROKEN_RECURSIVE_MACRO

/* Define if (void (*)(int)) casts are required for signals in C++ */
#undef SIGNAL_CAST

/* Define is sysinfo(int,char*,long) is available */
#undef HAVE_SYSINFO_3ARG

/* Define if linux sysinfo is available */
#undef HAVE_SYSINFO
