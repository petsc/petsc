#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.1 2001/04/05 19:33:48 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_win32 
#define PETSC_ARCH_NAME "win32_ms"
#define PETSC_HAVE_WIN32
#define HAVE_LIMITS_H
#define HAVE_STDLIB_H 
#define HAVE_STRING_H 
#define HAVE_SEARCH_H
#define HAVE_IO_H

#define HAVE_SYS_STAT_H

#define PETSC_HAVE_STD_COMPLEX

#define PETSC_STDCALL __stdcall
#define PETSC_USE_FORTRAN_MIXED_STR_ARG

#define PETSC_HAVE_FORTRAN_CAPS 

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_RAND
#define PETSC_CANNOT_START_DEBUGGER
#define HAVE_CLOCK

#define PETSC_HAVE_GET_USER_NAME
#define SIZEOF_VOID_P 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define PETSC_USE_NT_TIME
#define PETSC_HAVE_NO_GETRUSAGE

#define PETSC_HAVE_F90_H "f90impl/f90_win32.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_win32.c"

#define MISSING_SIGBUS
#define MISSING_SIGQUIT
#define MISSING_SIGSYS

#define HAVE__ACCESS
#define HAVE__GETCWD
#define HAVE__SLEEP
#define PETSC_USE_NARGS
#define PETSC_HAVE_IARG_COUNT_PROGNAME

#endif
