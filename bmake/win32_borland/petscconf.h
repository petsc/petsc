#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.4 2001/03/26 19:11:22 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_win32 
#define PETSC_ARCH_NAME "win32"
#define HAVE_LIMITS_H
#define HAVE_STDLIB_H 
#define HAVE_STRING_H 
#define HAVE_SEARCH_H
#define HAVE_IO_H
#define HAVE_GETCWD

#define PETSC_HAVE_TEMPLATED_COMPLEX
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

#define MISSING_SIGBUS
#define MISSING_SIGQUIT
#define MISSING_SIGSYS

#define HAVE_DOS_H
#define HAVE_ACCESS

#endif

