#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.26 2001/03/02 23:01:54 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_alpha
#define PETSC_ARCH_NAME "alpha"
#define PETSC_USE_GETCLOCK

#define PETSC_USE_KBYTES_FOR_SIZE
#define HAVE_POPEN
#define HAVE_LIMITS_H
#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H 
#define HAVE_STDLIB_H 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME  
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME  
#define PETSC_HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define HAVE_SYS_UTSNAME_H
#define HAVE_STRINGS_H
#define PETSC_USE_DBX_DEBUGGER
#define HAVE_SYS_RESOURCE_H

#define SIZEOF_VOID_P 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define PETSC_USE_DYNAMIC_LIBRARIES 1
#define PETSC_USE_NONEXECUTABLE_SO 1

#define PETSC_NEED_SOCKET_PROTO
#define HAVE_MACHINE_ENDIAN_H

#define PETSC_NEED_KILL_FOR_DEBUGGER
#define PETSC_USE_PID_FOR_DEBUGGER


#define PETSC_HAVE_F90_H "f90impl/f90_alpha.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_alpha.c"

#endif
