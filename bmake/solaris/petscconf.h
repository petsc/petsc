#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.30 2000/11/28 17:26:31 bsmith Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_solaris 
#define PETSC_ARCH_NAME "solaris"

#define HAVE_POPEN
#define PETSC_USE_CTABLE
#define HAVE_LIMITS_H
#define HAVE_STROPTS_H 
#define HAVE_SEARCH_H 
#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H
#define HAVE_STDLIB_H
#define HAVE_UNISTD_H 
#define HAVE_DRAND48 
#define HAVE_GETCWD
#define HAVE_SLEEP

#define HAVE_SYS_TIME_H
#define HAVE_SYS_SYSTEMINFO_H
#define HAVE_SYSINFO_3ARG
#define HAVE_SUNMATH_H
#define PETSC_HAVE_SUNMATHPRO
#define PETSC_HAVE_STD_COMPLEX

#define PETSC_HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_DOUBLES_ALIGNED
#define HAVE_DOUBLE_ALIGN_MALLOC

#define HAVE_MEMALIGN
#define PETSC_USE_DBX_DEBUGGER
#define HAVE_SYS_RESOURCE_H

#define HAVE_SYS_PROCFS_H
#define PETSC_USE_PROCFS_FOR_SIZE
#define HAVE_FCNTL_H
#define SIZEOF_VOID_P  4
#define SIZEOF_INT    4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#define WORDS_BIGENDIAN 1

#define PETSC_USE_DYNAMIC_LIBRARIES 1
#define PETSC_HAVE_RTLD_GLOBAL 1

#define HAVE_SYS_TIMES_H

#define PETS_PREFER_DCOPY_FOR_MEMCPY

#define PETSC_HAVE_F90_H "f90impl/f90_solaris.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_solaris.c"

#define HAVE_UCBPS

#define PETSC_HAVE_SOLARIS_STYLE_FPTRAP
#endif

