#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.12 2001/09/05 15:35:23 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_rs6000
#define PETSC_ARCH_NAME "rs6000"
#define PETSC_USE_READ_REAL_TIME

#define HAVE_POPEN
#define HAVE_LIMITS_H
#define HAVE_STROPTS_H 
#define HAVE_SEARCH_H 
#define HAVE_PWD_H 
#define HAVE_STDLIB_H
#define HAVE_STRING_H 
#define HAVE_STRINGS_H 
#define HAVE_MALLOC_H 
#define _POSIX_SOURCE
#define HAVE_DRAND48  
#define HAVE_GETCWD

#define HAVE_GETDOMAINNAME  
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 
#endif
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME 
#if !defined(_XOPEN_SOURCE_EXTENDED)
#define _XOPEN_SOURCE_EXTENDED 1
#endif
#if !defined(_ALL_SOURCE)
#define _ALL_SOURCE
#endif
#define HAVE_DOUBLE_ALIGN_MALLOC
#define PETSC_PREFER_BZERO
#define HAVE_READLINK
#define HAVE_MEMMOVE

#define PETSC_USE_DBX_DEBUGGER
#define HAVE_SYS_RESOURCE_H
#define SIZEOF_VOID_P 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define WORDS_BIGENDIAN 1
#define PETSC_NEED_SOCKET_PROTO
#define PETSC_HAVE_ACCEPT_SIZE_T

#define PETSC_HAVE_SLEEP_RETURNS_EARLY
#define PETSC_USE_KBYTES_FOR_SIZE
#define PETSC_USE_A_FOR_DEBUGGER

#define PETSC_HAVE_F90_H "f90impl/f90_rs6000.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_rs6000.c"

#define PETSC_HAVE_RS6000_STYLE_FPTRAP
#endif
