#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.7 2000/11/28 17:26:31 bsmith Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_linux
#define PETSC_ARCH_NAME "linux"

#define HAVE_POPEN
#define HAVE_LIMITS_H
#define HAVE_PWD_H 
#define HAVE_MALLOC_H 
#define HAVE_STRING_H 
#define HAVE_GETDOMAINNAME
#define HAVE_DRAND48 
#define HAVE_UNAME 
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_STDLIB_H
#define HAVE_UNISTD_H
#define HAVE_GETCWD
#define HAVE_SLEEP
#define HAVE_SYS_PARAM_H
#define HAVE_SYS_STAT_H

#define PETSC_HAVE_FORTRAN_UNDERSCORE 
#define PETSC_HAVE_TEMPLATED_COMPLEX

#define HAVE_READLINK
#define HAVE_MEMMOVE

#define HAVE_DOUBLE_ALIGN_MALLOC
#define HAVE_MEMALIGN
#define HAVE_SYS_RESOURCE_H
#define SIZEOF_VOID_P 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define BITS_PER_BYTE 8

#if defined(fixedsobug)
#define PETSC_USE_DYNAMIC_LIBRARIES 1
#define PETSC_HAVE_RTLD_GLOBAL 1
#endif

#define MISSING_SIGSYS

#endif
