#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.7 2001/02/09 19:40:53 bsmith Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_linux
#define PETSC_ARCH_NAME "linux"

#define HAVE_SLEEP
#define HAVE_SYS_WAIT_H 1
#define TIME_WITH_SYS_TIME 1
#define PETSC_HAVE_FORTRAN_UNDERSCORE 1
#define HAVE_DRAND48 1
#define HAVE_GETCWD 1
#define HAVE_GETHOSTNAME 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_MEMMOVE 1
#define HAVE_RAND 1
#define HAVE_READLINK 1
#define HAVE_REALPATH 1
#define HAVE_SIGACTION 1
#define HAVE_SIGNAL 1
#define HAVE_SIGSET 1
#define HAVE_SOCKET 1
#define HAVE_STRSTR 1
#define HAVE_UNAME 1
#define HAVE_FCNTL_H 1
#define HAVE_LIMITS_H 1
#define HAVE_MALLOC_H 1
#define HAVE_PWD_H 1
#define HAVE_SEARCH_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_STROPTS_H 1
#define HAVE_SYS_PROCFS_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_UNISTD_H 1
#define PETSC_HAVE_LIBNSL 1
#define HAVE_PARAM_H
#define HAVE_SYS_STAT_H

#define PETSC_USE_KBYTES_FOR_SIZE
#define HAVE_POPEN
#define HAVE_GETDOMAINNAME  
#define PETSC_USE_DBX_DEBUGGER

#define SIZEOF_VOID_P 8
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define PETSC_NEED_SOCKET_PROTO

#define PETSC_NEED_KILL_FOR_DEBUGGER
#define PETSC_USE_PID_FOR_DEBUGGER
#define PETSC_HAVE_TEMPLATED_COMPLEX

#define PETSC_HAVE_F90_H "f90impl/f90_alpha.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_alpha.c"

#define MISSING_SIGSYS

#endif
