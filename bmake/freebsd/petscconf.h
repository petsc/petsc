#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.22 2001/03/14 18:14:28 balay Exp $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_freebsd
#define PETSC_ARCH_NAME "freebsd"

#define HAVE_POPEN
#define HAVE_LIMITS_H
#define HAVE_PWD_H 
#define HAVE_STDLIB_H 
#define HAVE_STRING_H 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME 
#define HAVE_UNISTD_H  
#define HAVE_UNAME 
#define HAVE_SYS_TIME_H
#define HAVE_GETCWD
#define HAVE_SLEEP

#define BITS_PER_BYTE 8

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define PETSC_HAVE_TEMPLATED_COMPLEX

#define PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE
#define PETSC_HAVE_FORTRAN_UNDERSCORE

#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)
#define HAVE_VPRINTF_CHAR
#endif
#define HAVE_SYS_RESOURCE_H
#define SIZEOF_VOID_P 4
#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8

#define PETSC_USE_DYNAMIC_LIBRARIES 1
#define MISSING_DREAL
#define PETSC_HAVE_COMPILER_ATTRIBTE_CHECKING

#endif
