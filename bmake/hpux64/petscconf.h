#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.4 2000/09/27 19:08:40 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_hpux 
#define PETSC_ARCH_NAME "hpux"

#define HAVE_LIMITS_H
#define HAVE_STDLIB_H 
#define HAVE_PWD_H 
#define HAVE_MALLOC_H 
#define HAVE_STRING_H 
#define _POSIX_SOURCE 
#define _INCLUDE_POSIX_SOURCE
#define HAVE_DRAND48 
#define _INCLUDE_XOPEN_SOURCE 
#define _INCLUDE_XOPEN_SOURCE_EXTENDED 
#define _INCLUDE_HPUX_SOURCE 
#define HAVE_GETDOMAINNAME 
#define HAVE_SYS_TIME_H
#define HAVE_UNISTD_H 
#define HAVE_UNAME
#define HAVE_GETCWD
#define HAVE_SYS_PARAM_H
#define HAVE_SYS_STAT_H

#if defined(USING_ACC_FOR_CXX)
#define PETSC_HAVE_NONSTANDARD_COMPLEX_H "complex"
#else
#define PETSC_HAVE_TEMPLATED_COMPLEX
#endif

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define PETSC_USE_XDB_DEBUGGER

#define HAVE_SYS_RESOURCE_H

#define HAVE_CLOCK
#define SIZEOF_VOID_P 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define WORDS_BIGENDIAN 1
#define PETSC_NEED_SOCKET_PROTO

#define PETSC_HAVE_FORTRAN_UNDERSCORE

#define HAVE_SLEEP
#define PETSC_NEED_DEBUGGER_NO_SLEEP
#define PETSC_HAVE_NO_GETRUSAGE
#define PETSC_USE_LARGEP_FOR_DEBUGGER

#define PETSC_HAVE_F90_H "f90impl/f90_hpux.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_hpux.c"

#endif

