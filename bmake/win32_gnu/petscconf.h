#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.25 2001/06/20 21:20:06 buschelm Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_win32_gnu
#define PETSC_ARCH_NAME "win32_gnu"

#define HAVE_GETCWD
#define HAVE_POPEN
#define HAVE_LIMITS_H
#define HAVE_SLOW_NRM2
#define HAVE_SEARCH_H
#define HAVE_PWD_H
#define HAVE_STRING_H
#define HAVE_GETDOMAINNAME 
#define HAVE_UNISTD_H
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME
#define HAVE_MALLOC_H
#define HAVE_STDLIB_H
#define HAVE_UNISTD_H
#define HAVE_SYS_TIME_H
#define PETSC_NEEDS_GETTIMEOFDAY_PROTO
#define HAVE_SLEEP
#define HAVE_SYS_STAT_H

#define PETSC_HAVE_FORTRAN_UNDERSCORE 
#define PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define HAVE_RAND
#define HAVE_DOUBLE_ALIGN_MALLOC

#define PETSC_CANNOT_START_DEBUGGER
#define HAVE_SYS_RESOURCE_H

#define PETSC_HAVE_GET_USER_NAME
#define SIZEOF_VOID_P 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define PETSC_USE_NT_TIME

#define MISSING_SIGSYS

#ifdef PETSC_USE_MAT_SINGLE
#  define PETSC_MEMALIGN 16
#  define PETSC_HAVE_SSE "gccsse.h"
#endif

#define PETSC_PRINTF_FORMAT_CHECK(a,b) __attribute__ ((format (printf, a,b)))
 
#endif
